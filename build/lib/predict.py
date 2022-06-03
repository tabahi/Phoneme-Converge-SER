

import PhonemeSER
'''
pip install sklearn pydub wavio
'''


def main():
    # Make sure an already trained model file is available.
    model_file = "data/model_EmoD.RAVD.IEMO.Shem.DEMo.MSPI_single-random0.1_5_2_-0.5_16I32I64I12816D32D64D12860.0250.010.65_0.pkl"
    
    test_wav = "263771femaleprotagonist.wav"

    multi_classifiers_results = PhonemeSER.model_predict_wav_file(model_file, test_wav)
    print(multi_classifiers_results)

if __name__ == '__main__':
    main()

