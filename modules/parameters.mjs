const global = {
  ep_steps: 200,

  mb_len: 128,
  buffer_len: 2e4,
  discount: 0.95,

  get steps_before_training() {
    return 1 * this.mb_len
  },

  actorTauInitial: 0.01,
  criticTauInitial: 0.01,
  tauDecay: 1,
  actorTauMin: 0.01,
  criticTauMin: 0.01,

  obs_noise: 0.01,
  // lr_alpha: 0.1,
  // training_episodes: 10,

  g: 0.004,
  rRatio: 0.4,
  arc_display: 0.5,
  drag: 0.995,
  torque_mag: 0.002,
  omega_lim: 0.5,

  noise_sigma_initial: 0.3,
  noise_theta: 0.99,
  noise_bumper: 0.1,

  noise_decay: 0.9997,
  noise_min: 0.1
}

export default global
