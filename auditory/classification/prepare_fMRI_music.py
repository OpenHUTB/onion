# 根据 fMRI 实验的数据划分音频的训练集和测试集（测试集和验证集一样），
import librosa
import numpy as np
import os
import random
import shutil
from hparams import hparams

# 读取.mat文件
import scipy.io as scio

import shutil

from hparams import hparams



# 解压.tar.gz文件到当前文件夹
def extract_tar_files(tar_file):
    cur_dir, tar_file_name = os.path.split(tar_file);
    # tar_file_name = tar_file.split('/')[-1]
    des_dir = os.path.join(cur_dir, tar_file_name.split('.')[0])
    # 如果存在文件夹，则删除
    if os.path.exists(des_dir):
        shutil.rmtree(des_dir)  # 删除任意文件夹，可以不为空
        os.mkdir(des_dir)
    else:
        os.mkdir(des_dir)

    import tarfile
    f = tarfile.open(tar_file)
    names = f.getnames()
    for name in names:
        f.extract(name, path=cur_dir)
    return 0


def get_genre(hparams):
    return hparams.genres

def load_list(list_name, hparams):
    with open(os.path.join(hparams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()

    return file_names

def get_item(hparams, genre):
    return librosa.util.find_files(hparams.dataset_path + '/' + str(genre))


def readfile(file_name, hparams):
    y, sr = librosa.load(file_name, hparams.sample_rate)
    return y, sr


def change_pitch_and_speed(data):
    y_pitch_speed = data.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed


def change_pitch(data, sr):
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def value_aug(data):
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    y_aug = y_aug * dyn_change
    return y_aug


def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise


def hpss(data):
    y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'))
    return y_harmonic, y_percussive


def shift(data):
    return np.roll(data, 1600)


def stretch(data, rate=1):
    input_length = len(data)
    streching = librosa.effects.time_stretch(data, rate)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching


def change_speed(data):
    y_speed = data.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed


def melspectrogram(file_name, hparams):
    y, sr = librosa.load(os.path.join(hparams.dataset_path, file_name), hparams.sample_rate)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)

    mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T

    return mel_S


def resize_array(array, length):
    resize_array = np.zeros((length, array.shape[1]))
    if array.shape[0] >= length:
        resize_array = array[:length]
    else:
        resize_array[:array.shape[0]] = array
    return resize_array


# 根据 fMRI 实验的测试和验证数据集划分以及顺序生成 用于深度网络分类的训练集、测试集（验证集）和验证集
def prepare_data():
    # 为了复现结果
    random.seed(13)
    # extract_tar_files(os.path.join(hparams.data_path, 'genres.tar.gz'))
    # fMRI 中的训练集、验证集、测试集
    ExpTrnOrder = scio.loadmat('../music_genre_fMRI/ExpTrnOrder.mat')
    ExpTrnOrder = ExpTrnOrder['TrnOrd']
    train_rows, train_cols = ExpTrnOrder.shape
    ExpValOrder = scio.loadmat('../music_genre_fMRI/ExpValOrder.mat');
    ExpValOrder = ExpValOrder['ValOrd']
    val_rows, val_cols = ExpValOrder.shape
    workspace_dir = '/data3/dong/data/brain/auditory'
    run_data_dir = os.path.join(workspace_dir, 'run_dataset')
    if not os.path.exists(run_data_dir):
        os.mkdir(run_data_dir)
    else:
        shutil.rmtree(run_data_dir)
        os.mkdir(run_data_dir)

    # 为各个流派的的音乐名创建目录
    all_genres = hparams.genres
    for i in range(len(all_genres)):
        cur_genres_dir = os.path.join(run_data_dir, all_genres[i])
        if not os.path.exists(cur_genres_dir):
            os.mkdir(cur_genres_dir)
    src_wave_dir = os.path.join(workspace_dir, 'process_music_fMRI', 'genres_wav15s')

    for j in range(val_cols):
        for i in range(val_rows):
            cur_wave_name = ExpValOrder[i][j][0]
            src_wave_path = os.path.join(src_wave_dir, cur_wave_name)
            cur_genres_name = cur_wave_name.split('.')[0]
            dst_wave_path = os.path.join(run_data_dir, cur_genres_name, ('Test_run-%02d-%02d_' + cur_wave_name) % ((j+1), (i+1)))  # 存在相同的音频文件，fMRI记录时候用了多次
            shutil.copy(src_wave_path, dst_wave_path)
            with open(os.path.join(run_data_dir, "valid_list.txt"), 'a') as f:
                f.writelines(dst_wave_path + "\n")
            with open(os.path.join(run_data_dir, "test_list.txt"), 'a') as f:
                f.writelines(dst_wave_path + "\n")

    for j in range(train_cols):
        for i in range(train_rows):
            cur_wave_name = ExpTrnOrder[i][j][0]
            src_wave_path = os.path.join(src_wave_dir, cur_wave_name)
            cur_genres_name = cur_wave_name.split('.')[0]
            dst_wave_path = os.path.join(run_data_dir, cur_genres_name, ('Train_run-%02d_'  + cur_wave_name) % (j+1))
            shutil.copy(src_wave_path, dst_wave_path)
            with open(os.path.join(run_data_dir, "train_list.txt"), 'a') as f:
                f.writelines(dst_wave_path + "\n")


# 对测试集进行10中数据增强，测试集怎么也有了？
def augment_audio():
    print('Augmentation')
    genres = get_genre(hparams)
    list_names = ['train_list.txt']  # 只对训练集进行扩充（只在train_list.txt进行添加，还没有真正增强）
    for list_name in list_names:
        # file_names = load_list(list_name, hparams)
        with open(os.path.join(hparams.run_dataset_path, list_name)) as f:
            file_names = f.read().splitlines()
        with open(os.path.join(hparams.run_dataset_path, list_name), 'w') as f:
            for i in file_names:
                f.writelines(os.path.join(hparams.run_dataset_path, i + '\n'))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'a.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'b.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'c.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'd.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'e.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'f.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'g.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'h.wav' + '\n')))
                f.writelines(os.path.join(hparams.run_dataset_path, i.replace('.wav', 'i.wav' + '\n')))

    # 这是对所有的音频进行增强，包括了测试集
    for genre in genres:
        item_list = librosa.util.find_files(hparams.run_dataset_path + '/' + str(genre))
        # item_list = get_item(hparams, genre)
        for file_name in item_list:
            y, sr = readfile(file_name, hparams)
            data_noise = add_noise(y)
            data_roll = shift(y)
            data_stretch = stretch(y)
            pitch_speed = change_pitch_and_speed(y)
            pitch = change_pitch(y, hparams.sample_rate)
            speed = change_speed(y)
            value = value_aug(y)
            y_harmonic, y_percussive = hpss(y)
            y_shift = shift(y)

            (save_path, save_name) = os.path.split(file_name)
            # save_path = os.path.join(file_name.split(genre + '.')[0])
            # save_name = genre + '.' + file_name.split(genre + '.')[1]
            print(save_name)

            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'a.wav')), data_noise,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'b.wav')), data_roll,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'c.wav')), data_stretch,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'd.wav')), pitch_speed,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'e.wav')), pitch,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'f.wav')), speed,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'g.wav')), value,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'h.wav')), y_percussive,
                                     hparams.sample_rate)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.wav', 'i.wav')), y_shift,
                                     hparams.sample_rate)
        print('finished')


# 抽取特征
def extract_feature():
    print("Extracting Feature")
    list_names = ['train_list.txt', 'valid_list.txt', 'test_list.txt']

    for list_name in list_names:
        set_name = list_name.replace('_list.txt', '')
        with open(os.path.join(hparams.run_dataset_path, list_name)) as f:
            file_names = f.read().splitlines()
        # file_names = load_list(list_name, hparams)

        for file_name in file_names:
            feature = melspectrogram(file_name, hparams)
            feature = resize_array(feature, hparams.feature_length)

            # Data Arguments
            num_chunks = feature.shape[0] / hparams.num_mels
            data_chuncks = np.split(feature, num_chunks)

            for idx, i in enumerate(data_chuncks):
                save_path = os.path.join(hparams.feature_fMRI_path, set_name, file_name.split('/')[-2])
                save_name = file_name.split('/')[-1].split('.wav')[0] + str(idx) + ".npy"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(os.path.join(save_path, save_name), i.astype(np.float32))
                print(os.path.join(save_path, save_name))

    print('finished feature extraction')


def main():
    prepare_data()
    augment_audio()
    extract_feature()


if __name__ == '__main__':
    main()
