"""Тренировочная сеть"""
# Базовый пример обучаемой нейронной сети
from time import process_time
from pathlib import Path
import tensorflow as tf
import numpy as np
from .model import create_model


def train(hparams):
    """Постройте и обучите модель, как указано в hparams"""

    ckptsdir = str(Path(hparams.logdir, "ckpts"))

    try:
        tf.gfile.MakeDirs(ckptsdir)
    except Exception as e:
        print("Exception encountered when trying to make directory %s" % ckptsdir)
        print("Error %s" % e)
        quit()

    # построение графиков обучения и оценки
    train_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.TRAIN)
    eval_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    with train_tuple.graph.as_default():
        initializer = tf.global_variables_initializer()

    with eval_tuple.graph.as_default():
        local_initializer = tf.local_variables_initializer()

    if hparams.logging:
        summary_writer = tf.summary.FileWriter(hparams.logdir,
                                               train_tuple.graph,
                                               max_queue=25,
                                               flush_secs=30)

    train_tuple.session.run([initializer])

    start_time = process_time()

    # инициализация набора обучающих данных
    train_tuple.session.run([train_tuple.iterator.initializer])
    train_tuple.graph.finalize()

    profile_next_step = False
    eval_step = 100
    patience = 0
    max_patience = hparams.num_keep_ckpts-1
    best_eval_loss = np.Inf
    best_step = -1

    # Обучение, пока набор данных не выдаст ошибку (в конце num_epochs)
    while patience < max_patience:
        step_time = []
        try:
            curr_time = process_time()
            # ПРИМЕЧАНИЕ. Чтобы включить профилирование, удалите False в этом утверждении.
            if False:  # profile_next_step and hparams.logging:
                # запуск профайлинга
                _, train_loss, global_step, summary = train_tuple.model.train_with_profile(train_tuple.session, summary_writer)
                profile_next_step = False
            else:
                _, train_loss, global_step, summary = train_tuple.model.train(train_tuple.session)
            step_time.append(process_time() - curr_time)

            # обобщение в ходе процесса обучения
            if global_step == 1 and hparams.logging:
                summary_writer.add_summary(summary, global_step)
            if global_step % 15 == 0:
                if hparams.logging:
                    summary_writer.add_summary(summary, global_step)
                print("Step: %d, Training Loss: %4.4f, Avg Sec/Step: %2.2f" % (global_step, train_loss, np.mean(step_time)))

            if global_step % eval_step == 0:
                step_time = []
                profile_next_step = True
                # одна проверка
                checkpoint_path = train_tuple.model.saver.save(train_tuple.session,
                                                               ckptsdir+"/ckpt",
                                                               global_step=global_step)
                eval_tuple.model.saver.restore(eval_tuple.session, checkpoint_path)
                eval_tuple.session.run([eval_tuple.iterator.initializer, local_initializer])
                while True:
                    try:
                        eval_loss, eval_acc, _, eval_summary, _ = eval_tuple.model.eval(eval_tuple.session)
                    except tf.errors.OutOfRangeError:
                        print("Step: %d, Eval Loss: %4.4f, Eval Accuracy: %1.4f" % (global_step,
                                                                              eval_loss,
                                                                              eval_acc))
                        if eval_loss < best_eval_loss:
                            patience = 0
                            best_eval_loss = eval_loss
                            best_step = global_step
                        else:
                            patience += 1
                            print("Patience: %d" % patience)

                        if hparams.logging:
                            summary_writer.add_summary(eval_summary, global_step)
                        break

        except tf.errors.OutOfRangeError:
            print("- End of Training -")
            break

    # Окончение обучения
    summary_writer.close()
    print("Best validation loss: %3.3f at step %d" % (best_eval_loss, best_step))
    print("Total Training Time: %4.2f" % (process_time() - start_time))
