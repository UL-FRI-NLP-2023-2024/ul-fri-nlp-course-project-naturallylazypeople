from trainers.trainer_fft import TrainerFFT

class LoRaTrainer(TrainerFFT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add any additional initialization here

    # Override methods as needed
