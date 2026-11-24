// Parser examples recorded from mlx-vlm 5fe604cfce2a55db69916245e0e8a7b5aebaaddf.
let referenceToolFixtures = #"""
    [
      {
        "format": "json",
        "text": "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\",\"days\":2}}</tool_call>",
        "calls": [
          {
            "name": "weather",
            "arguments": {
              "city": "Paris",
              "days": 2
            }
          }
        ]
      },
      {
        "format": "json",
        "text": "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"text\":\"é 🏳️\",\"items\":[true,null,3],\"nested\":{\"key\":\"<tool_call> literal\"}}}</tool_call>",
        "calls": [
          {
            "name": "lookup",
            "arguments": {
              "text": "é 🏳️",
              "items": [
                true,
                null,
                3
              ],
              "nested": {
                "key": "<tool_call> literal"
              }
            }
          }
        ]
      },
      {
        "format": "lfm2",
        "text": "<|tool_call_start|>[weather(city='Paris', days=2)]<|tool_call_end|>",
        "calls": [
          {
            "name": "weather",
            "arguments": {
              "city": "Paris",
              "days": 2
            }
          }
        ]
      },
      {
        "format": "lfm2",
        "text": "<|tool_call_start|>[lookup(text='hello', items=[True, None, 3], nested={'key': 'value'})]<|tool_call_end|>",
        "calls": [
          {
            "name": "lookup",
            "arguments": {
              "text": "hello",
              "items": [
                true,
                null,
                3
              ],
              "nested": {
                "key": "value"
              }
            }
          }
        ]
      },
      {
        "format": "lfm2",
        "text": "<|tool_call_start|>[weather(city='Paris'), weather(city='Tokyo')]<|tool_call_end|>",
        "calls": [
          {
            "name": "weather",
            "arguments": {
              "city": "Paris"
            }
          },
          {
            "name": "weather",
            "arguments": {
              "city": "Tokyo"
            }
          }
        ]
      }
    ]
    """#
