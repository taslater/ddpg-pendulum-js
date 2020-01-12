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

const pendulum = new Pendulum()
let step = 0
let episode = 0

async function draw() {
  ctx.clearRect(0, 0, wh, wh)
  if (step >= global.ep_steps) {
    step = 0
    episode++
    pendulum.reset()
    window.requestAnimationFrame(draw)
  }
  pendulum.show(ctx, wh)
  pendulum.update(episode <= global.training_episodes)
  if (episode > global.training_episodes) {
    await pendulum.ddpg.train(step)
  }
  step++
  window.requestAnimationFrame(draw)
}

draw()
