import global from "./parameters.mjs"

export default function showPendulum(ctx, wh, animationState) {
  const theta = animationState.theta,
    torque = animationState.torque,
    action = animationState.action,
    noise = animationState.noise

  const center = Math.floor(0.5 * wh)
  ctx.clearRect(0, 0, wh, wh)

  ctx.lineCap = "butt"
  ctx.strokeStyle = `rgb(100,100,100)`
  ctx.lineWidth = 1

  ctx.beginPath()
  ctx.moveTo(center, Math.floor(0.05 * wh))
  ctx.lineTo(center, Math.floor(0.95 * wh))
  ctx.stroke()

  const _theta = theta - 0.5 * Math.PI
  const v = Math.round(100 * (1 - Math.abs(theta) / Math.PI)) + 100

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

  const _theta_torque = _theta - global.arc_display * torque

  let sorted_angles = [_theta, _theta_torque].sort((a, b) => a - b)
  ctx.beginPath()
  ctx.arc(center, center, Math.floor(wh * global.rRatio), ...sorted_angles)
  ctx.stroke()

  ctx.strokeStyle = "blue"
  // ctx.lineWidth = 10
  sorted_angles = [_theta, _theta - global.arc_display * action].sort(
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
    _theta - global.arc_display * torque,
    _theta - global.arc_display * torque + global.arc_display * noise
  ].sort((a, b) => a - b)
  ctx.beginPath()
  ctx.arc(center, center, 0.8 * wh * global.rRatio, ...sorted_angles)
  ctx.stroke()

  ctx.lineCap = "round"
  ctx.strokeStyle = `rgb(100,100,100)`
  ctx.lineWidth = 3

  const cw_lim = _theta - global.arc_display,
    cw_lim_sn = Math.sin(cw_lim),
    cw_lim_csn = Math.cos(cw_lim),
    ccw_lim = _theta + global.arc_display,
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

  // ctx.strokeStyle = "black"
  // ctx.lineCap = "round"
  // ctx.lineWidth = 0.045 * wh
  // ctx.beginPath()
  // ctx.moveTo(center, center)
  // ctx.lineTo(center, center)
  // ctx.stroke()
}
