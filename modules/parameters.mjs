const global = {
  ep_steps: 300,

  mb_len: 128,
  buffer_len: 2e4,
  discount: 0.995,

  get steps_before_training() {
    return Math.max(4 * this.mb_len, 1.1 * this.mb_len)
  },

  actorTauInitial: 0.01,
  criticTauInitial: 0.01,
  tauDecay: 0.99995,
  actorTauMin: 0.002,
  criticTauMin: 0.002,

  obs_noise: 2e-3,
  // lr_alpha: 0.1,
  // training_episodes: 10,

  g: 0.003,
  rRatio: 0.4,
  arc_display: 0.5,
  drag: 0.995,
  torque_mag: 0.0015,
  omega_lim: 0.5,

  noise_sigma_initial: 0.2,
  noise_sigma_min: 0.02,
  noise_decay: 0.9998,
  noise_theta: 0.98,
  noise_bumper: 0.1
}

export default global
