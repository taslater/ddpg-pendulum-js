// import { Pendulum } from "./modules/pendulum.mjs"
import showPendulum from "./modules/showPendulum.mjs"
import global from "./modules/parameters.mjs"

// tf.setBackend("webgl")

let wh
let pendulumCanvas = document.getElementById("pendulum-canvas")
let ctx = pendulumCanvas.getContext("2d")

// const pendulum = new Pendulum()
// const ddpg_worker = new Worker("./modules/ddpg_worker.js")
const agent_worker = new Worker("./modules/agent_worker.js")
// const animationQueue = []

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

agent_worker.addEventListener("message", e => {
  requestAnimationFrame(() => {
    showPendulum(ctx, wh, e.data)
    agent_worker.postMessage("next frame")
  })
})

// debugger
agent_worker.postMessage({
  global
})
