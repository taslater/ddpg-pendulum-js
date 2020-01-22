this.window = this

importScripts(
  "./ziggurat.js",
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js",
  "./nn_models.js"
)

tf.enableProdMode()
tf.setBackend("cpu")

let theta, torque, action, noise, noise_sigma
let global, state_len, targetActor
const ddpg_worker = new Worker("./ddpg_worker.js")
let ep_step = 0,
  episode = 0
let zig = new Ziggurat()
let initialized = false
let looper

onmessage = e => {
  const msg = e.data
  if (!initialized && msg.hasOwnProperty("global")) {
    initialize(msg.global)
    run()
  } else if (initialized && msg == "next frame") {
    run()
  }
}

ddpg_worker.addEventListener("message", e => {
  setTimeout(() => {
    // targetActor.setWeights(e.data)
    tf.tidy(() => {
      const wts = targetActor.getWeights()
      const new_wts = e.data
      for (let i = 0; i < wts.length; i++) {
        wts[i] = tf.tensor(new_wts[i], wts[i].shape)
      }
      targetActor.setWeights(wts)
    })
    noise_sigma *= global.noise_decay
    if (noise_sigma < global.noise_sigma_min) {
      noise_sigma = noise_sigma_min
    }
    // console.log(e.data)
  }, 0)
})

function initialize(_global) {
  global = _global

  noise_sigma = global.noise_sigma_initial
  reset()
  state_len = state().length

  stateBuffer = new ArrayBuffer(4 * state_len)
  stateView = new Float32Array(stateBuffer)

  targetActor = Actor(false, state_len)

  initialized = true

  ddpg_worker.postMessage({
    settings: { global, state_len, actorWeights: targetActor.getWeights() }
  })
}

function run() {
  if (ep_step >= global.ep_steps) {
    ep_step = 0
    episode++
    reset()
  } else {
    ep_step++
  }
  ddpg_worker.postMessage({
    experience: Object.assign({}, update())
  })
  postMessage(Object.assign({}, animationState()))
}

function animationState() {
  return { theta, torque, action, noise }
}

function reset() {
  theta = 2 * Math.PI * (Math.random() - 0.5)
  omega = 0
  noise = noise_sigma * (0.5 - Math.random())
  action = 0
  updateTorque()
}

function reward() {
  return 1 - Math.abs(theta / Math.PI)
  // return (
  //   -Math.abs(theta / Math.PI) - Math.abs((0.1 * torque) / global.torque_mag)
  // )
}

function state() {
  const csn = Math.cos(theta),
    sn = Math.sin(theta)
  return [csn, sn, 10 * omega]
}

function update() {
  const s0 = state().slice()
  const r0 = reward()
  updateAction()
  updateNoise()
  updateTorque()
  updatePhysics()
  const s1 = state().slice()
  const r1 = reward()
  return { s0, a: action, r: r1 - r0, s1 }
}

function updateAction() {
  const s = state().slice()
  for (let i = 0; i < state_len; i++) {
    stateView[i] = s[i]
  }
  action = tf.tidy(() => {
    return targetActor
      .predict(tf.tensor(stateView, [1, state_len], "float32"), {
        batchSize: 1
      })
      .dataSync()[0]
  })
}

function updateNoise() {
  noise *= global.noise_theta
  noise += noise_sigma * zig.nextGaussian()

  if (action + noise > 1) {
    noise = 1 - global.noise_bumper - action
  } else if (action + noise < -1) {
    noise = -1 + global.noise_bumper - action
  }
}

function updateTorque() {
  torque = action + noise
}

function updatePhysics() {
  omega *= global.drag
  omega += global.torque_mag * torque + global.g * Math.sin(theta)
  if (Math.abs(omega) > global.omega_lim) {
    omega *= global.omega_lim / Math.abs(omega)
  }
  theta += omega
  if (theta > Math.PI) {
    theta -= 2 * Math.PI
  } else if (theta < -Math.PI) {
    theta += 2 * Math.PI
  }
}
