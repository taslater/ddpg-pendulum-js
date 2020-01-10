import { createCanvas } from "./modules/canvas.mjs"
import { Pendulum } from "./modules/pendulum.mjs"
import { getActor } from "./modules/nn_models.js"

const w = 600,
  h = 600

let pendulumCanvas = createCanvas(
  "pendulumCanvas",
  document.getElementById("main"),
  w,
  h,
  "black"
)

pendulumCanvas.ctx.lineCap = "round"

const pendulum = new Pendulum(w, h)
const state_len = pendulum.state.slice().length
const actor = getActor(false, state_len)

function draw() {
  pendulumCanvas.ctx.clearRect(0, 0, w, h)
  pendulum.show(pendulumCanvas.ctx)
  const action = tf.tidy(() => {
    return actor
      .predict(tf.tensor(pendulum.state.slice(), [1, state_len]), {
        batchSize: 1
      })
      .dataSync()[0]
  })
  pendulum.update(action)
  window.requestAnimationFrame(draw)
}

draw()
