import SwiftUI

/// A flat, opaque surface shared by the message and image composers.
struct ComposerSurface<Content: View>: View {
    #if os(iOS)
        @Environment(\.horizontalSizeClass) private var sizeClass
    #endif
    var content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(16)
            .background(.background, in: RoundedRectangle(cornerRadius: 24))
            .overlay {
                RoundedRectangle(cornerRadius: 24)
                    .strokeBorder(.quaternary)
            }
            .frame(maxWidth: 760)
            .padding(.horizontal, horizontalPadding)
            .padding(.top, 8)
            .padding(.bottom, 18)
            .frame(maxWidth: .infinity)
    }

    private var horizontalPadding: CGFloat {
        #if os(iOS)
            sizeClass == .compact ? 12 : 24
        #else
            24
        #endif
    }
}
