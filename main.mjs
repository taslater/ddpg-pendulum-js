import { createCanvas } from "./modules/canvas.mjs"
import { Pendulum } from "./modules/pendulum.mjs"
import global from "./modules/parameters.mjs"

let pendulumCanvas = createCanvas(
  "pendulumCanvas",
  document.getElementById("main"),
  global.wh,
  global.wh,
  "black"
)

const pendulum = new Pendulum()
let step = 0
let episode = 0

async function draw() {
  if (step >= global.ep_steps) {
    step = 0
    episode++
    pendulum.reset()
  }
  pendulumCanvas.ctx.clearRect(0, 0, global.wh, global.wh)
  pendulum.show(pendulumCanvas.ctx)
  pendulum.update(episode <= global.training_episodes)
  if (episode > global.training_episodes) {
    await pendulum.ddpg.train(step)
  }
  step++
  window.requestAnimationFrame(draw)
}

draw()
