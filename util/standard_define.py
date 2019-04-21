def is_satisfied_standard2(predict_list, right_location):
    ''' 标准2，预测允许前后2个距离单元内有物体，不管是否是虚假目标，有目标即可，且只允许在这几个距离单元内预测有目标 '''
    if right_location >= 2 and right_location <= 62:
        if predict_list[right_location] == 1 \
                or predict_list[right_location + 1] == 1 or \
                predict_list[right_location - 1] == 1 or \
                predict_list[right_location + 2] == 1 or \
                predict_list[right_location - 2] == 1:

            for i in range(right_location + 3, 64):
                if predict_list[i] == 1:
                    return False

            for i in range(right_location - 2):
                if predict_list[i] == 1:
                    return False

            return True
    else:
        if predict_list[right_location] == 1:
            return True
        else:
            return False


def is_satisfied_standard3(predict_list, right_location):
    ''' 标准2，预测允许前后2个距离单元内有物体，不管是否是虚假目标，有目标即可，且只允最多允许3个虚假目标 '''
    if right_location >= 2 and right_location <= 62:
        if predict_list[right_location] == 1 \
                or predict_list[right_location + 1] == 1 or \
                predict_list[right_location - 1] == 1 or \
                predict_list[right_location + 2] == 1 or \
                predict_list[right_location - 2] == 1:

            target_count = 0
            for i in range(0, 64):
                if predict_list[i] == 1:
                    target_count = target_count + 1

            if target_count > 5:
                return False

            return True
    else:
        if predict_list[right_location] == 1:
            return True
        else:
            return False
