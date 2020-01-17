import { Ziggurat } from "./ziggurat.mjs"
import global from "./parameters.mjs"

export class Pendulum {
  constructor() {
    this.action
    this.noise_mag = global.noise_mag_initial
    this.zig = new Ziggurat()
    this.reset()
    this.arc_display = 0.5 // radians
  }

  reset() {
    this.theta = 2 * Math.PI * (Math.random() - 0.5)
    // this.prev_theta = this.theta
    this.omega = 0
    this.noise = 0.25 * this.noise_mag * (0.5 - Math.random())
    // this.noise = 0
  }

  get reward() {
    return -Math.abs(this.theta)
    // return Math.cos(this.theta)
    // return -(
    //   this.theta * this.theta * this.theta * this.theta +
    //   Math.abs(this.omega) * this.omega * this.omega +
    //   this.torque ** 2
    // )
    // return -(
    //   Math.abs(this.theta) +
    // Math.abs(this.omega) +
    // 0.01 * Math.abs(this.torque)
    // )
  }

  get state() {
    const csn = Math.cos(this.theta),
      sn = Math.sin(this.theta)
    return [
      csn,
      sn,
      10 * this.omega

      // 0.5 - Math.abs(this.theta) / Math.PI,
      // Math.cos(this.theta - this.omega) - csn,
      // Math.sin(this.theta - this.omega) - sn,
      // this.theta > 0 ? 1 : 0,
      // sn > 0 ? 1 : -1,
      // this.omega > 0 ? 1 : -1
      // Math.abs(10 * this.omega)
    ]
  }

  update(action) {
    this.action = action
    const s0 = this.state.slice()
    // this.prev_theta = this.theta
    this.noise *= global.noise_theta
    this.noise += global.noise_sigma * this.zig.nextGaussian()
    // this.action = 0
    if (action !== 0) {
      this.noise_mag *= global.noise_decay
      if (this.noise_mag < global.noise_min) {
        this.noise_mag = global.noise_min
      }
      // tf.setBackend("cpu")
      // this.action = this.ddpg.getAction(s0)
    }
    // this.action *= 1 - 0.5 * this.noise_mag
    if (this.action + this.noise * this.noise_mag > 1) {
      this.noise = (1 - global.noise_bumper - this.action) / this.noise_mag
    } else if (this.action + this.noise * this.noise_mag < -1) {
      this.noise = (-(1 - global.noise_bumper) - this.action) / this.noise_mag
    }
    this.torque = this.action + this.noise * this.noise_mag

    const experience = { s0: s0, a: this.torque }
    const reward0 = this.reward
    this.omega +=
      global.torque_mag * this.torque + global.g * Math.sin(this.theta)
    this.omega *= global.drag
    if (Math.abs(this.omega) > global.omega_lim) {
      this.omega *= global.omega_lim / Math.abs(this.omega)
    }
    this.theta += this.omega
    if (this.theta > Math.PI) {
      this.theta -= 2 * Math.PI
    } else if (this.theta < -Math.PI) {
      this.theta += 2 * Math.PI
    }
    const reward1 = this.reward
    experience.r = reward1 - reward0
    // experience.r = reward1
    experience.s1 = this.state.slice()
    // this.ddpg.replay_buffer.add(Object.assign({}, experience))
    return Object.assign({}, experience)
  }

  show(ctx, wh) {
    const center = Math.floor(0.5 * wh)
    ctx.clearRect(0, 0, wh, wh)

    ctx.lineCap = "butt"
    ctx.strokeStyle = `rgb(100,100,100)`
    ctx.lineWidth = 1

    ctx.beginPath()
    ctx.moveTo(center, Math.floor(0.05 * wh))
    ctx.lineTo(center, Math.floor(0.95 * wh))
    ctx.stroke()

    const _theta = this.theta - 0.5 * Math.PI
    const v = Math.round(100 * (1 - Math.abs(this.theta) / Math.PI)) + 100

    ctx.lineCap = "round"
    ctx.strokeStyle = `rgb(${v},${v},${v})`
    ctx.lineWidth = 0.06 * wh

    ctx.beginPath()
    ctx.moveTo(center, center)
    ctx.lineTo(
      Math.floor(center * (1 + 2 * global.rRatio * Math.cos(_theta))),
      Math.floor(center * (1 + 2 * global.rRatio * Math.sin(_theta)))
    )
    ctx.stroke()

    ctx.lineCap = "butt"
    ctx.strokeStyle = "red"
    ctx.lineWidth = Math.floor(0.1 * wh * global.rRatio)

    const _theta_torque = _theta - this.arc_display * this.torque

    let sorted_angles = [_theta, _theta_torque].sort((a, b) => a - b)
    ctx.beginPath()
    ctx.arc(center, center, Math.floor(wh * global.rRatio), ...sorted_angles)
    ctx.stroke()

    ctx.strokeStyle = "blue"
    // ctx.lineWidth = 10
    sorted_angles = [_theta, _theta - this.arc_display * this.action].sort(
      (a, b) => a - b
    )
    ctx.beginPath()
    ctx.arc(
      center,
      center,
      Math.floor(0.9 * wh * global.rRatio),
      ...sorted_angles
    )
    ctx.stroke()

    ctx.strokeStyle = "green"
    // ctx.lineWidth = 10
    sorted_angles = [
      _theta - this.arc_display * this.torque,
      _theta -
        this.arc_display * this.torque +
        this.arc_display * this.noise * this.noise_mag
    ].sort((a, b) => a - b)
    ctx.beginPath()
    ctx.arc(center, center, 0.8 * wh * global.rRatio, ...sorted_angles)
    ctx.stroke()

    ctx.lineCap = "round"
    ctx.strokeStyle = `rgb(100,100,100)`
    ctx.lineWidth = 3

    const cw_lim = _theta - this.arc_display,
      cw_lim_sn = Math.sin(cw_lim),
      cw_lim_csn = Math.cos(cw_lim),
      ccw_lim = _theta + this.arc_display,
      ccw_lim_sn = Math.sin(ccw_lim),
      ccw_lim_csn = Math.cos(ccw_lim),
      lim_line_len = 0.3

    // clock-wise
    ctx.beginPath()
    ctx.moveTo(
      center * (1 + (1.8 + lim_line_len) * global.rRatio * cw_lim_csn),
      center * (1 + (1.8 + lim_line_len) * global.rRatio * cw_lim_sn)
    )
    ctx.lineTo(
      center * (1 + (1.8 - lim_line_len) * global.rRatio * cw_lim_csn),
      center * (1 + (1.8 - lim_line_len) * global.rRatio * cw_lim_sn)
    )
    ctx.stroke()

    // counter-clock-wise
    ctx.beginPath()
    ctx.moveTo(
      center * (1 + (1.8 + lim_line_len) * global.rRatio * ccw_lim_csn),
      center * (1 + (1.8 + lim_line_len) * global.rRatio * ccw_lim_sn)
    )
    ctx.lineTo(
      center * (1 + (1.8 - lim_line_len) * global.rRatio * ccw_lim_csn),
      center * (1 + (1.8 - lim_line_len) * global.rRatio * ccw_lim_sn)
    )
    ctx.stroke()

    const torque_sn = Math.sin(_theta_torque),
      torque_csn = Math.cos(_theta_torque)

    ctx.strokeStyle = `rgb(150,150,150)`
    // connect torque and noise arcs
    ctx.beginPath()
    ctx.moveTo(
      center * (1 + (1.8 + lim_line_len) * global.rRatio * torque_csn),
      center * (1 + (1.8 + lim_line_len) * global.rRatio * torque_sn)
    )
    ctx.lineTo(
      center * (1 + (1.8 - lim_line_len) * global.rRatio * torque_csn),
      center * (1 + (1.8 - lim_line_len) * global.rRatio * torque_sn)
    )
    ctx.stroke()

    ctx.strokeStyle = "black"
    ctx.lineCap = "round"
    ctx.lineWidth = 0.045 * wh
    ctx.beginPath()
    ctx.moveTo(center, center)
    ctx.lineTo(center, center)
    ctx.stroke()
  }
}
