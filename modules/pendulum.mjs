import { Ziggurat } from "./ziggurat.mjs"
import { DDPG } from "./ddpg.mjs"
import global from "./parameters.mjs"

export class Pendulum {
  constructor() {
    this.drag = 0.995
    this.torque_mag = 0.003
    this.action
    this.omega_lim = 0.5
    this.noise_sigma = 0.05
    this.noise_theta = 0.995
    this.noise_mag = 4
    this.noise_decay = 0.9997
    this.noise_min = 0.5
    this.zig = new Ziggurat()
    this.ddpg = new DDPG(this.state.length)
    this.reset()
    this.arc_display = 0.5 // radians
  }

  reset() {
    this.theta = 2 * Math.PI * (Math.random() - 0.5)
    this.omega = 0
    this.noise = this.noise_mag * (0.5 - Math.random())
    // this.noise = 0
    this.torque = 0
  }

  get reward() {
    // return -Math.abs(this.theta)
    // return Math.cos(this.theta)
    return -(
      this.theta * this.theta * this.theta * this.theta +
      0.1 * Math.abs(this.omega) * this.omega * this.omega +
      0.01 * this.torque ** 2
    )
    // return -(
    //   Math.abs(this.theta) +
    //   Math.abs(this.omega) +
    //   Math.abs(this.torque)
    // )
  }

  get state() {
    const csn = Math.cos(this.theta)
    const sn = Math.sin(this.theta)
    return [
      csn,
      sn,
      csn * csn,
      sn * sn,
      0.5 - Math.abs(this.theta) / Math.PI,
      // this.theta,
      10 * this.omega
    ]
  }

  update(initial) {
    const s0 = this.state.slice()
    this.noise *= this.noise_theta
    this.noise += this.noise_sigma * this.zig.nextGaussian()
    this.action = 0
    if (!initial) {
      this.noise_mag *= this.noise_decay
      if (this.noise_mag < this.noise_min) {
        this.noise_mag = this.noise_min
      }
      // tf.setBackend("cpu")
      this.action = tf.tidy(() => {
        return this.ddpg.targetActor
          .predict(tf.tensor(s0, [1, s0.length]), {
            batchSize: 1
          })
          .dataSync()[0]
      })
    }
    if (this.action + this.noise * this.noise_mag > 1) {
      this.noise = (0.8 - this.action) / this.noise_mag
    } else if (this.action + this.noise * this.noise_mag < -1) {
      this.noise = (-0.8 - this.action) / this.noise_mag
    }
    this.torque = this.action + this.noise * this.noise_mag

    const experience = { s0: s0, a: this.torque }
    const reward0 = this.reward
    this.omega +=
      this.torque_mag * this.torque + global.g * Math.sin(this.theta)
    this.omega *= this.drag
    if (Math.abs(this.omega) > this.omega_lim) {
      this.omega *= this.omega_lim / Math.abs(this.omega)
    }
    this.theta += this.omega
    if (this.theta > Math.PI) {
      this.theta -= 2 * Math.PI
    } else if (this.theta < -Math.PI) {
      this.theta += 2 * Math.PI
    }
    const reward1 = this.reward
    experience.r = reward1 - reward0
    experience.s1 = this.state.slice()
    this.ddpg.replay_buffer.add(Object.assign({}, experience))
  }

  show(ctx, wh) {
    // ctx.globalAlpha = 0.9
    ctx.lineCap = "round"
    ctx.strokeStyle = `rgb(30,30,30)`
    ctx.lineWidth = 3

    ctx.beginPath()
    ctx.moveTo(0.5 * wh, 0.05 * wh)
    ctx.lineTo(0.5 * wh, 0.95 * wh)
    ctx.stroke()

    const _theta = this.theta - 0.5 * Math.PI
    const v = Math.round(100 * (1 - Math.abs(this.theta) / Math.PI)) + 100

    ctx.lineCap = "round"
    ctx.strokeStyle = `rgb(${v},${v},${v})`
    ctx.lineWidth = 0.06 * wh

    ctx.beginPath()
    ctx.moveTo(0.5 * wh, 0.5 * wh)
    ctx.lineTo(
      0.5 * wh * (1 + 2 * global.rRatio * Math.cos(_theta)),
      0.5 * wh * (1 + 2 * global.rRatio * Math.sin(_theta))
    )
    ctx.stroke()

    ctx.lineCap = "butt"
    ctx.strokeStyle = "red"
    ctx.lineWidth = 0.1 * wh * global.rRatio

    const _theta_torque = _theta - this.arc_display * this.torque

    let sorted_angles = [_theta, _theta_torque].sort((a, b) => a - b)
    ctx.beginPath()
    ctx.arc(0.5 * wh, 0.5 * wh, wh * global.rRatio, ...sorted_angles)
    ctx.stroke()

    ctx.strokeStyle = "blue"
    // ctx.lineWidth = 10
    sorted_angles = [_theta, _theta - this.arc_display * this.action].sort(
      (a, b) => a - b
    )
    ctx.beginPath()
    ctx.arc(0.5 * wh, 0.5 * wh, 0.9 * wh * global.rRatio, ...sorted_angles)
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
    ctx.arc(0.5 * wh, 0.5 * wh, 0.8 * wh * global.rRatio, ...sorted_angles)
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
      0.5 * wh * (1 + (1.8 + lim_line_len) * global.rRatio * cw_lim_csn),
      0.5 * wh * (1 + (1.8 + lim_line_len) * global.rRatio * cw_lim_sn)
    )
    ctx.lineTo(
      0.5 * wh * (1 + (1.8 - lim_line_len) * global.rRatio * cw_lim_csn),
      0.5 * wh * (1 + (1.8 - lim_line_len) * global.rRatio * cw_lim_sn)
    )
    ctx.stroke()

    // counter-clock-wise
    ctx.beginPath()
    ctx.moveTo(
      0.5 * wh * (1 + (1.8 + lim_line_len) * global.rRatio * ccw_lim_csn),
      0.5 * wh * (1 + (1.8 + lim_line_len) * global.rRatio * ccw_lim_sn)
    )
    ctx.lineTo(
      0.5 * wh * (1 + (1.8 - lim_line_len) * global.rRatio * ccw_lim_csn),
      0.5 * wh * (1 + (1.8 - lim_line_len) * global.rRatio * ccw_lim_sn)
    )
    ctx.stroke()

    const torque_sn = Math.sin(_theta_torque),
      torque_csn = Math.cos(_theta_torque)

    ctx.strokeStyle = `rgb(150,150,150)`
    // connect torque and noise arcs
    ctx.beginPath()
    ctx.moveTo(
      0.5 * wh * (1 + (1.8 + lim_line_len) * global.rRatio * torque_csn),
      0.5 * wh * (1 + (1.8 + lim_line_len) * global.rRatio * torque_sn)
    )
    ctx.lineTo(
      0.5 * wh * (1 + (1.8 - lim_line_len) * global.rRatio * torque_csn),
      0.5 * wh * (1 + (1.8 - lim_line_len) * global.rRatio * torque_sn)
    )
    ctx.stroke()

    ctx.strokeStyle = "black"
    ctx.lineCap = "round"
    ctx.lineWidth = 0.045 * wh
    ctx.beginPath()
    ctx.moveTo(0.5 * wh, 0.5 * wh)
    ctx.lineTo(0.5 * wh, 0.5 * wh)
    ctx.stroke()
  }
}
