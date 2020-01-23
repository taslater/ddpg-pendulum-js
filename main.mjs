import showPendulum from "./modules/showPendulum.mjs"
import global from "./modules/parameters.mjs"

let wh
let pendulumCanvas = document.getElementById("pendulum-canvas")
let ctx = pendulumCanvas.getContext("2d")

const agent_worker = new Worker("./modules/agent_worker.js")
const ddpg_worker = new Worker("./modules/ddpg_worker.js")

function resizeCanvas() {
  let navbarHeight = document.getElementById("navbar").offsetHeight
  document.body.style.paddingTop = navbarHeight + "px"
  wh = Math.min(window.innerHeight - navbarHeight, window.innerWidth)
  pendulumCanvas.width = wh
  pendulumCanvas.height = wh
  pendulumCanvas.style.top = navbarHeight + "px"
}
resizeCanvas()
window.onresize = resizeCanvas

agent_worker.addEventListener("message", e => {
  if (e.data.hasOwnProperty("state_len")) {
    ddpg_worker.postMessage({
      settings: {
        global,
        state_len: e.data.state_len,
        actorWeights: e.data.actorWeights
      }
    })
  } else if (
    e.data.hasOwnProperty("animationState") &&
    e.data.hasOwnProperty("experience")
  ) {
    ddpg_worker.postMessage({ experience: e.data.experience })
    requestAnimationFrame(() => {
      showPendulum(ctx, wh, e.data.animationState)
      agent_worker.postMessage("next frame")
    })
  }
})

ddpg_worker.addEventListener("message", e => {
  if (e.data.hasOwnProperty("newActorWts")) {
    agent_worker.postMessage({ newActorWts: e.data.newActorWts })
  }
})

agent_worker.postMessage({
  global
})
