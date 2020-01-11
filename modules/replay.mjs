import global from "./parameters.mjs"
import { shuffle } from "./shuffle.mjs"

export class ReplayBuffer {
  constructor() {
    this.data = []
    this.buffer_len = global.buffer_len
    this.mb_len = global.mb_len
  }
  add(obj) {
    this.data.push(obj)
    if (this.data.length > global.buffer_len) {
      this.data.splice(0, this.data.length - global.buffer_len)
    }
  }
  sample() {
    // return this.sample_without_replacement()
    return this.sample_from_recent_memory()
  }
  sample_from_recent_memory() {
    const mb_arr = []
    for (let i = 0; i < global.mb_len; i++) {
      const rand_idx = Math.floor(this.data.length * Math.sqrt(Math.random()))
      mb_arr.push(this.data[rand_idx])
    }
    return mb_arr
  }
  sample_without_replacement() {
    let idx_arr = [...Array(global.mb_len).keys()]
    idx_arr = shuffle(idx_arr)
    const mb_arr = []
    for (let i = 0; i < global.mb_len; i++) {
      mb_arr.push(this.data[idx_arr[i]])
    }
    return mb_arr
  }
}
