import { Pendulum } from "./modules/pendulum.mjs"
import global from "./modules/parameters.mjs"
// importScripts("./parameters.js")

// tf.setBackend("webgl")

let wh
let pendulumCanvas = document.getElementById("pendulum-canvas")
let ctx = pendulumCanvas.getContext("2d")

function resizeCanvas() {
  let navbarHeight = document.getElementById("navbar").offsetHeight
  document.body.style.paddingTop = navbarHeight + "px"
  wh = Math.min(window.innerHeight - navbarHeight, window.innerWidth)
  pendulumCanvas.width = wh
  // setTimeout(function() {
  pendulumCanvas.height = wh
  // }, 0)
  pendulumCanvas.style.top = navbarHeight + "px"
}
resizeCanvas()
window.onresize = resizeCanvas

window.addEventListener("DOMContentLoaded", event => {
  document
    .getElementById("settings-btn")
    .addEventListener("click", toggleOverlay)
})

function toggleOverlay() {
  const divs = document.getElementsByClassName("blurbspace")
  for (let div of divs) {
    div.style.display = div.style.display == "none" ? "block" : "none"
  }
  const controlsDiv = document.getElementById("controls-div")
  controlsDiv.style.display = divs[0].style.display == "none" ? "block" : "none"
}

let ep_step = 0,
  experience,
  episode = 0
const pendulum = new Pendulum()
const ddpg_worker = new Worker("./modules/ddpg_worker.js")
ddpg_worker.postMessage({
  settings: {
    state_len: pendulum.state.length,
    global: Object.assign({}, global)
  }
})

ddpg_worker.addEventListener("message", e => {
  const action = e.data.action
  experience = pendulum.update(action)

  requestAnimationFrame(() => {
    ddpg_worker.postMessage({ experience: experience, ep_step: ep_step })
    pendulum.show(ctx, wh)
  })
  if (ep_step >= global.ep_steps) {
    ep_step = 0
    episode++
    pendulum.reset()
  }
  ep_step++
})

requestAnimationFrame(() => {
  pendulum.show(ctx, wh)
})
experience = pendulum.update(0)
ddpg_worker.postMessage({ experience: experience, ep_step: ep_step })
ep_step++
