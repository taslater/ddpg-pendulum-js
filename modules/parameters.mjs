export default {
  ep_steps: 200,

  mb_len: 128,
  buffer_len: 2e4,
  discount: 0.95,

  actorTauInitial: 0.01,
  criticTauInitial: 0.01,
  tauDecay: 0.9999,
  actorTauMin: 0.001,
  criticTauMin: 0.001,

  obs_noise: 0.001,
  // lr_alpha: 0.1,
  // training_episodes: 10,

  g: 0.004,
  rRatio: 0.4,
  drag: 0.995,
  torque_mag: 0.002,
  omega_lim: 0.5,

  noise_sigma: 0.3,
  noise_theta: 0.99,
  noise_bumper: 0.1,

  noise_mag_initial: 1,
  noise_decay: 0.9997,
  noise_min: 0.1
}
