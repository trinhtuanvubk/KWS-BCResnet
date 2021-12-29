from prepare_data.download_GSC import download_GSC
from prepare_data.build_json_GSC import GSC_json
from config.config import Config

def GSC(config):

    print('DOWNLOAD GOOGLE SPEECH COMMANDS')
    download_GSC(config)

    print('BUILD JSON FOR GOOGLE SPEECH COMMANDS')
    GSC_json(config)

if __name__=="__main__":
    config = Config()
    GSC(config=config)