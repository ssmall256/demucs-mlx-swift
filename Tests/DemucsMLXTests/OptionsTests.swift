import Foundation
import Testing

@testable import DemucsMLX

@Test func defaults() {
  let options = SeparationOptions()
  #expect(options.shifts == 1)
  #expect(options.overlap == 0.25)
  #expect(options.attention == .fp16)
  #expect(DemucsModel.allCases.count == 8)
}
@Test func invalidOptions() {
  var options = SeparationOptions()
  options.overlap = 1
  #expect(throws: DemucsError.self) { try Separator.validate(options) }
  options.overlap = Float.nan
  #expect(throws: DemucsError.self) { try Separator.validate(options) }
  options.overlap = 0.25
  options.batchSize = 0
  #expect(throws: DemucsError.self) { try Separator.validate(options) }
  options.batchSize = 1
  options.segmentSeconds = Double.infinity
  #expect(throws: DemucsError.self) { try Separator.validate(options) }
}
@Test func pythonShiftSequence() {
  var generator = ShiftRandom(seed: 481)
  let actual = (0..<8).map { _ in generator.offset(maximum: 22050) }
  #expect(actual == [15507, 13921, 21580, 4557, 15114, 21020, 10096, 15181])
}
