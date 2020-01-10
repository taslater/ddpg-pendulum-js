import { Ziggurat } from "./ziggurat.mjs"

export class Pendulum {
  constructor(w, h) {
    this.w = w
    this.h = h
    this.g = 0.005
    this.theta = 2 * Math.PI * (Math.random() - 0.5)
    this.omega = 0
    this.drag = 0.995
    this.torque = 0
    this.torque_mag = 0.002
    this.noise = 0
    this.noise_sigma = 0.1
    this.noise_theta = 0.98
    this.noise_lim = 1
    this.r = 0.4 * w
    this.zig = new Ziggurat()
  }

  get reward() {
    return -(
      this.theta ** 4 +
      0.1 * Math.abs(this.omega) ** 3 +
      0.01 * this.torque ** 2
    )
    // return -(
    //   Math.abs(this.theta) +
    //   Math.abs(this.omega) +
    //   Math.abs(this.torque)
    // )
  }

  get state() {
    return [
      Math.cos(this.theta),
      Math.sin(this.theta),
      this.theta / Math.PI,
      10 * this.omega
    ]
  }
  update(action) {
    this.omega *= this.drag
    this.noise *= this.noise_theta
    this.noise += this.noise_sigma * this.zig.nextGaussian()
    if (Math.abs(this.noise) > this.noise_lim) {
      this.noise *= this.noise_lim / Math.abs(this.noise)
    }
    this.torque = action + this.noise
    if (Math.abs(this.torque) > 1) {
      this.torque /= Math.abs(this.torque)
    }
    this.omega += this.torque_mag * this.torque + this.g * Math.sin(this.theta)
    this.theta += this.omega
  }
  show(ctx) {
    ctx.strokeStyle = "white"
    ctx.lineWidth = 40
    ctx.beginPath()
    ctx.moveTo(0.5 * this.w, 0.5 * this.h)
    ctx.lineTo(
      0.5 * this.w + this.r * Math.sin(this.theta),
      0.5 * this.h - this.r * Math.cos(this.theta)
    )
    ctx.stroke()

    ctx.strokeStyle = "red"
    ctx.lineWidth = 10
    const _theta = this.theta - 0.5 * Math.PI
    const sorted_angles = [
      _theta,
      _theta - 0.5 * (this.noise + this.torque)
    ].sort((a, b) => a - b)
    ctx.beginPath()
    ctx.arc(0.5 * this.w, 0.5 * this.h, this.r, ...sorted_angles)
    ctx.stroke()

    ctx.strokeStyle = "black"
    ctx.lineWidth = 30
    ctx.beginPath()
    ctx.moveTo(0.5 * this.w, 0.5 * this.h)
    ctx.lineTo(0.5 * this.w, 0.5 * this.h)
    ctx.stroke()
  }
}
