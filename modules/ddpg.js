class DDPG {
  constructor(replay_len) {
    this.replay = new ReplayBuffer(replay_len)
    this.actor
    this.critic
  }
}
