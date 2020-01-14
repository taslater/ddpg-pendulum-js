import { Pendulum } from "./modules/pendulum.mjs"
import global from "./modules/parameters.mjs"

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

const pendulum = new Pendulum()
let step = 0
let episode = 0
let done_training = true

async function draw() {
  ctx.clearRect(0, 0, wh, wh)
  if (step >= global.ep_steps) {
    step = 0
    episode++
    pendulum.reset()
  }
  pendulum.show(ctx, wh)
  pendulum.update(episode <= global.training_episodes)
  if (episode > global.training_episodes && done_training) {
    done_training = false
    await pendulum.ddpg.train(step).then((done_training = true))
  }
  step++
  window.requestAnimationFrame(draw)
}

draw()
