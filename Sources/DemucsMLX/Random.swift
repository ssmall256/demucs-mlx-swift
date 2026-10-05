// CPython-compatible integer-seeded MT19937 and getrandbits rejection sampling.
// Matching offsets makes Swift/Python comparisons reproducible without changing
// the default one-shift inference behavior.
struct ShiftRandom {
  var state = Array(repeating: UInt32(0), count: 624)
  var index = 624
  init(seed: UInt64) {
    var x: UInt32 = 19_650_218
    state[0] = x
    for i in 1..<624 {
      x = 1_812_433_253 &* (x ^ (x >> 30)) &+ UInt32(i)
      state[i] = x
    }
    let key: [UInt32] =
      seed >> 32 == 0
      ? [UInt32(truncatingIfNeeded: seed)] : [UInt32(truncatingIfNeeded: seed), UInt32(seed >> 32)]
    var i = 1
    var j = 0
    for _ in 0..<max(624, key.count) {
      state[i] =
        (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) &* 1_664_525)) &+ key[j] &+ UInt32(j)
      i += 1
      j += 1
      if i >= 624 {
        state[0] = state[623]
        i = 1
      }
      if j >= key.count { j = 0 }
    }
    for _ in 0..<623 {
      state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) &* 1_566_083_941)) &- UInt32(i)
      i += 1
      if i >= 624 {
        state[0] = state[623]
        i = 1
      }
    }
    state[0] = 0x8000_0000
  }
  mutating func next() -> UInt32 {
    if index >= 624 {
      for i in 0..<624 {
        let y = (state[i] & 0x8000_0000) | (state[(i + 1) % 624] & 0x7fff_ffff)
        state[i] = state[(i + 397) % 624] ^ (y >> 1) ^ (y & 1 == 0 ? 0 : 0x9908_b0df)
      }
      index = 0
    }
    var y = state[index]
    index += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9d2c_5680
    y ^= (y << 15) & 0xefc6_0000
    y ^= y >> 18
    return y
  }
  mutating func offset(maximum: Int) -> Int {
    let n = UInt32(maximum + 1)
    let bits = 32 - n.leadingZeroBitCount
    while true {
      let v = next() >> (32 - bits)
      if v < n { return Int(v) }
    }
  }
}
