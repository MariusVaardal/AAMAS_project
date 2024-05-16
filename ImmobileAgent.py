from SimpleTagAgent import SimpleTagAgent

class ImmobileAgent(SimpleTagAgent):
    def __init__(self, name, num_adversaries, num_landmarks) -> None:
        super().__init__(name, num_adversaries, num_landmarks)
    
    def get_action(self) -> list:
        return 0