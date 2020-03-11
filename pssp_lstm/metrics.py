"""Метрики для оценки."""

import tensorflow as tf


def streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Вычисляет confusion matrix.

    Этот метод создает локальные переменные для отслеживания статистики матрицы путаницы
    потоков данных.
    Args:
        labels: the ground truth labels, a Tensor одинаковой формы с прогнозами
        predictions: прогнозируемые значения, Tensor следующей формы (?,)
        num_classes: число классов для confusion matrix
        weights: вес каждого прогноза
    Returns:
        confusion: A k x k Tensor отображает confusion matrix, где
            столбцы представляют прогнозируемую метку, а строки представляют
            истинный ярлык
        update_op: Операция, которая обновляет значения в confusion_matrix
            соответственно.
    """

    _confusion = tf.confusion_matrix(labels=labels,
                                     predictions=predictions,
                                     num_classes=num_classes,
                                     weights=weights,
                                     name="_confusion")

    # аккумулятор для confusion matrix
    confusion = tf.get_local_variable(name="confusion",
                                      shape=[num_classes, num_classes],
                                      dtype=tf.int32,
                                      initializer=tf.zeros_initializer)

    # обновление op
    update_op = confusion.assign(confusion + _confusion)

    confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                 [1, num_classes, num_classes, 1])

    tf.summary.image('confusion_matrix', confusion_image)

    return confusion, update_op


def cm_summary(confusion, num_classes):
    """
    Returns:
        confusion_summary: Общее описание confusion matrix как image
    """
    confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                 [1, num_classes, num_classes, 1])

    confusion_summary = tf.summary.image('confusion_matrix', confusion_image)
    return confusion_summary
