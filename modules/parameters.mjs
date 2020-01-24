const global = {
  ep_steps: 300,

  mb_len: 64,
  buffer_len: 2e4,
  discount: 0.95,

  get steps_before_training() {
    return Math.max(8 * this.mb_len, 1.1 * this.mb_len, 900)
  },

  actorTauInitial: 0.01,
  criticTauInitial: 0.01,
  tauDecay: 1,
  actorTauMin: 0.01,
  criticTauMin: 0.01,

  // obs_noise: 1e-3,
  training_critic_obs_noise: 1e-3,
  target_critic_obs_noise: 0,
  training_actor_obs_noise: 1e-3,
  target_actor_obs_noise: 0,

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
  noise_decay: 1 - 0.0001,
  noise_theta: 1 - 0.03,
  noise_bumper: 0.1
}

export default global
