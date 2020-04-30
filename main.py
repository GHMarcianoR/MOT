from core import Core
from time import time
import argparse


def arguments_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_files', type=str, default='mot17/test/')
    parser.add_argument('--name', type=str, default='MOT17-06-DPM')
    parser.add_argument('--draw_objects', type=bool, default=True)
    parser.add_argument('--show_objects', type=bool, default=False)
    parser.add_argument('--record_video', type=bool, default=True)
    parser.add_argument('--train', type=bool, required=True)
    parser.add_argument('--min_iou', type=float, default=0.5)
    parser.add_argument('--num_max_miss', type=int, default=3)
    parser.add_argument('--min_conf', type=float, default=0)
    return parser.parse_args()


def write_results(name_, min_iou, num_max_miss, total_time_, results_):
    log_file = open('evaluation_results/' + str(name_) + '.txt', 'a')
    log_file.write('Tempo de Processamento: {:.2f}\n'.format(total_time_))
    log_file.write('Valor minimo de IoU: {}\n'.format(min_iou))
    log_file.write('Numero maximo de falhas (miss): {}\n\n'.format(num_max_miss))
    log_file.write(results_)
    log_file.write('\n')
    log_file.write('################################################################################################')
    log_file.write('\n\n')
    log_file.close()


def start_process(path_, name_, d_ob=True, s_ob=False, r_vd=True,
                  train=False, min_iou=.5, m_miss=1,
                  conf_v=0):
    draw_ = (d_ob, s_ob, r_vd)
    start = time()
    path_ = path_ + '/' + name_ + '/'

    core = Core(path_, draw_, min_iou,
                m_miss, conf_v, train)

    core.main_loop()
    end = time()
    if train:
        write_results(name_,
                      min_iou,
                      m_miss,
                      end - start,
                      core.train_results())
    else:
        core.test_results(name_)


if __name__ == '__main__':
    arguments = arguments_parser()
    start_process(arguments.path_files,
                  arguments.name,
                  arguments.draw_objects,
                  arguments.show_objects,
                  arguments.record_video,
                  arguments.train,
                  arguments.min_iou,
                  arguments.num_max_miss)
