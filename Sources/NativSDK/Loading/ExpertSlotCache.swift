import Darwin
import Foundation
import MLX

/// Experimental final-weight cache. Slots are reused only after the caller has
/// evaluated its previous expert result; speculative reads never overwrite pins.
final class ExpertSlotCache: @unchecked Sendable {
    typealias Key = ExpertWeightStore.Key
    typealias Layer = ExpertWeightStore.Layer
    private final class Allocation {
        let pointer: UnsafeMutableRawPointer
        let capacity: Int
        init(_ count: Int) throws {
            let page = Int(getpagesize())
            capacity = (count + page - 1) / page * page
            var address: UnsafeMutableRawPointer?
            guard posix_memalign(&address, page, capacity) == 0, let address else {
                throw ExpertWeightStore.invalid("unable to allocate expert slots")
            }
            pointer = address
        }
        deinit { free(pointer) }
    }
    private struct Entry {
        let slot: Int
        var ready: Bool
        var age: UInt64
    }
    private let condition = NSCondition()
    private var entries: [Key: Entry] = [:]
    private var freeSlots: [Int]
    private var pins = Set<Key>()
    private var age: UInt64 = 0
    private let allocations: [String: Allocation]
    // These strong roots prevent donation of externally mutable weight storage.
    private let roots: [MLXArray]
    let arrays: [String: MLXArray]
    let count: Int
    let byteCount: Int

    static func compatible(_ layers: [Layer]) -> Bool {
        guard let first = layers.first, !first.tensors.isEmpty else { return false }
        return layers.allSatisfy { layer in
            Set(layer.tensors.keys) == Set(first.tensors.keys)
                && layer.tensors.allSatisfy { name, tensor in
                    let other = first.tensors[name]!
                    return tensor.shape == other.shape && tensor.dtype == other.dtype
                        && tensor.bytesPerExpert == other.bytesPerExpert
                }
        }
    }

    init(layers: [Layer], budget: Int) throws {
        let first = layers[0]
        let bytes = first.tensors.values.reduce(0) { $0 + $1.bytesPerExpert }
        let page = Int(getpagesize())
        // Include page padding in the budget. Need room for the whole routed batch.
        count = max(1, (budget - first.tensors.count * (page - 1)) / bytes)
        freeSlots = Array((0 ..< count).reversed())
        var buffers: [String: Allocation] = [:]
        var arrayRoots: [MLXArray] = []
        var views: [String: MLXArray] = [:]
        for (name, tensor) in first.tensors {
            let buffer = try Allocation(count * tensor.bytesPerExpert)
            let root = MLXArray(
                rawPointer: buffer.pointer, [buffer.capacity / tensor.dtype.size],
                dtype: tensor.dtype
            ) { [buffer] in
                withExtendedLifetime(buffer) {}
            }
            arrayRoots.append(root)
            views[name] = root[..<(count * tensor.bytesPerExpert / tensor.dtype.size)]
                .reshaped([count] + tensor.shape)
            buffers[name] = buffer
        }
        allocations = buffers
        roots = arrayRoots
        arrays = views
        byteCount = buffers.values.reduce(0) { $0 + $1.capacity }
    }

    func ready(_ keys: [Key]) -> Set<Key> {
        condition.lock()
        defer { condition.unlock() }
        return Set(keys.filter { entries[$0]?.ready == true })
    }

    /// Must be called only after the previous batch's scatter has completed.
    func begin(_ keys: [Key]) throws {
        guard Set(keys).count <= count else {
            throw ExpertWeightStore.invalid("expert slot capacity is smaller than batch")
        }
        condition.lock()
        defer { condition.unlock() }
        pins = Set(keys)
    }

    /// Called only after the GPU has finished using the selected slots.
    func releasePins() {
        condition.lock()
        pins.removeAll(keepingCapacity: true)
        condition.unlock()
    }

    /// No MLX work runs on reader threads. They write only to disjoint unpinned slots.
    @discardableResult
    func read(
        _ key: Key, tensors: [String: ExpertWeightStore.Tensor],
        handles: [URL: FileHandle], speculative: Bool
    ) throws -> Int? {
        condition.lock()
        let slot: Int
        while true {
            if let entry = entries[key] {
                if entry.ready {
                    age &+= 1
                    entries[key] = Entry(slot: entry.slot, ready: true, age: age)
                    condition.unlock()
                    return entry.slot
                }
            } else if let available = freeSlots.popLast() {
                slot = available
                break
            } else if let victim = entries.filter({ $0.value.ready && !pins.contains($0.key) })
                .min(by: { $0.value.age < $1.value.age })
            {
                slot = victim.value.slot
                entries[victim.key] = nil
                break
            }
            if speculative {
                condition.unlock()
                return nil
            }
            guard entries.values.contains(where: { !$0.ready }) else {
                condition.unlock()
                throw ExpertWeightStore.invalid("all expert slots are in flight")
            }
            // An unrelated speculative read may temporarily occupy the last
            // slot. Once it completes it can be evicted for this demand batch.
            // Recheck the key too: another reader may have started it meanwhile.
            condition.wait()
        }
        age &+= 1
        entries[key] = Entry(slot: slot, ready: false, age: age)
        condition.unlock()
        do {
            let ordered = tensors.sorted {
                $0.value.location(for: key.expert).offset
                    < $1.value.location(for: key.expert).offset
            }
            let first = ordered[0].value.location(for: key.expert)
            var cursor = first.offset
            let contiguous = ordered.allSatisfy { _, tensor in
                let location = tensor.location(for: key.expert)
                defer { cursor += UInt64(tensor.bytesPerExpert) }
                return location.url == first.url && location.offset == cursor
            }
            if contiguous {
                var vectors = ordered.map { name, tensor in
                    iovec(
                        iov_base: allocations[name]!.pointer + slot * tensor.bytesPerExpert,
                        iov_len: tensor.bytesPerExpert)
                }
                try vectors.withUnsafeMutableBufferPointer { buffer in
                    var current = 0
                    var done = 0
                    while current < buffer.count {
                        let n = preadv(
                            handles[first.url]!.fileDescriptor, buffer.baseAddress! + current,
                            Int32(buffer.count - current), off_t(first.offset) + off_t(done))
                        if n < 0 && errno == EINTR { continue }
                        guard n > 0 else {
                            throw ExpertWeightStore.invalid("failed expert record read")
                        }
                        done += n
                        var consumed = n
                        while current < buffer.count && consumed >= buffer[current].iov_len {
                            consumed -= buffer[current].iov_len
                            current += 1
                        }
                        if consumed > 0 {
                            buffer[current].iov_base! += consumed
                            buffer[current].iov_len -= consumed
                        }
                    }
                }
            } else {
                for (name, tensor) in tensors.sorted(by: {
                    $0.value.location(for: key.expert).offset
                        < $1.value.location(for: key.expert).offset
                }) {
                    let location = tensor.location(for: key.expert)
                    let fd = handles[location.url]!.fileDescriptor
                    let pointer = allocations[name]!.pointer + slot * tensor.bytesPerExpert
                    var done = 0
                    while done < tensor.bytesPerExpert {
                        let readCount = pread(
                            fd, pointer + done, tensor.bytesPerExpert - done,
                            off_t(location.offset) + off_t(done))
                        if readCount < 0 && errno == EINTR { continue }
                        guard readCount > 0 else {
                            throw ExpertWeightStore.invalid(
                                "failed slot read for layer \(key.layer), expert \(key.expert)")
                        }
                        done += readCount
                    }
                }
            }
            // Generic projections may carry one NVFP4 scale per expert. Keep
            // the same validation as the host cache before exposing those bytes.
            for name in ["weight_global_scale", "weight_scale_2"] {
                guard let tensor = tensors[name] else { continue }
                let pointer = allocations[name]!.pointer + slot * tensor.bytesPerExpert
                let value: Float
                switch tensor.dtype {
                case .float32:
                    value = Float(
                        bitPattern: UInt32(littleEndian: pointer.loadUnaligned(as: UInt32.self)))
                case .float16:
                    value = Float(
                        Float16(
                            bitPattern: UInt16(littleEndian: pointer.loadUnaligned(as: UInt16.self))
                        ))
                case .bfloat16:
                    value = Float(
                        bitPattern: UInt32(
                            UInt16(littleEndian: pointer.loadUnaligned(as: UInt16.self))) << 16)
                default: value = .nan
                }
                guard value.isFinite, value > 0 else {
                    throw ExpertWeightStore.invalid(
                        "invalid NVFP4 global scale at layer \(key.layer), expert \(key.expert)")
                }
            }
            condition.lock()
            entries[key]!.ready = true
            condition.broadcast()
            condition.unlock()
            return slot
        } catch {
            condition.lock()
            entries[key] = nil
            freeSlots.append(slot)
            condition.broadcast()
            condition.unlock()
            throw error
        }
    }

    func selected(_ keys: [Key]) throws -> [Int] {
        condition.lock()
        defer { condition.unlock() }
        return try keys.map {
            guard let entry = entries[$0], entry.ready else {
                throw ExpertWeightStore.invalid("expert slot read did not complete")
            }
            return entry.slot
        }
    }
}
