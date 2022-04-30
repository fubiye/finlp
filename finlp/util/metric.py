class EntityMetrics(object):
    """用于实体级别评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, std_tags, predict_tags):
        """
        初始化对照文件中的标签列表、预测文件的标签列表、以及
        :param std_tags:
        :param predict_tags:
        """
        #将按句的标签列表转化成按字的标签列表 如 [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.std_tags = flatten_lists(std_tags)     #将已知标签和预测的标签拼成列表
        self.predict_tags = flatten_lists(predict_tags)

        #计数器
        self.std_entity_counter = self.count_entity_dict(self.std_tags)  #标准结果的各实体个数
        self.predict_entity_counter = self.count_entity_dict(self.predict_tags)  #预测结果的各实体个数
        print("标准各实体个数", self.std_entity_counter)
        print("预测各实体个数", self.predict_entity_counter)

        self.std_entity_number = self.count_entity(self.std_tags)    #标准结果的实体总个数
        # self.predict_entity_number = self.count_entity(self.predict_tags)  #预测结果的实体总个数
        print("标准实体数", self.std_entity_number)
        # print("预测实体数", self.predict_entity_number)

        self.corrent_entity_number = self.count_correct_entity()
        print("正确的实体", self.corrent_entity_number)

        self.entity_set = set(self.std_entity_counter)
        print("实体集合", self.entity_set)

        # 计算精确率
        self.precision_scores = self.cal_precision()
        print("各个实体的准确率", self.precision_scores)

        # 计算召回率
        self.recall_scores = self.cal_recall()
        print("各个实体的召回率", self.recall_scores)

        # 计算F1分数
        self.f1_scores = self.cal_f1()
        print("各个实体的f1值", self.f1_scores)

        # 计算加权均值
        self.wighted_average = self._cal_wighted_average()
        print("各项指标的加权均值", self.wighted_average)

    def cal_precision(self):
    #计算每类实体的准确率
        precision_scores = {}
        #某实体正确的个数  /  预测中某实体所有的个数
        for entity in self.entity_set:                                                              #这里计算结果可能有问题
            precision_scores[entity] = self.corrent_entity_number.get(entity, 0) / max(1e-10, self.predict_entity_counter[entity])
        return precision_scores

    def cal_recall(self):
    #计算每类尸体的召回率
        recall_scores ={}
        for entity in self.entity_set:
            recall_scores[entity] = self.corrent_entity_number.get(entity, 0) / max(1e-10, self.std_entity_counter[entity])
        return recall_scores


    def cal_f1(self):
    #计算f1值
        f1_scores = {}
        for entity in self.entity_set:
            p, r = self.precision_scores[entity], self.recall_scores[entity]
            f1_scores[entity] = 2 * p * r / (p + r + 1e-10)
        return f1_scores

    def report_scores(self):
        """
        将结果用表格的形式打印出来
        :return:
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))
        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个实体的p, r, f
        for entity in self.entity_set:
            print(row_format.format(
                entity,
                self.precision_scores[entity],
                self.recall_scores[entity],
                self.f1_scores[entity],
                self.std_entity_counter[entity]   #这部分是support的值
            ))
        #计算并打印平均值
        avg_metrics = self._cal_wighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            self.std_entity_number
        ))

    def _cal_wighted_average(self):
        #计算加权均值

        weighted_average = {}
        total = self.std_entity_number  #标准实体的总数

        #计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for entity in self.entity_set:
            size = self.std_entity_counter[entity]  #标准结果各个实体的个数
            weighted_average['precision'] += self.precision_scores[entity] * size
            weighted_average['recall'] += self.recall_scores[entity] * size
            weighted_average['f1_score'] += self.f1_scores[entity] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def count_entity_dict(self, tag_list):
        """
        计算每个实体对应的个数，注意BME、BE、S结构才为实体，其余均不计入 B-company E-company算实体
        :param tag_list:
        :return:
        """
        enti_dict = {}
        flag = 0  # 初始状态设置为0
        for tag in tag_list:
            if 'B-' in tag and flag == 0:  #当B-出现时，将状态变为1
                flag = 1
            if 'E-' in tag and flag == 1: #E前有B则可以判定为一个实体
                flag = 0  #状态回归初始
                tag = tag[2:]
                if tag not in enti_dict:
                    enti_dict[tag] = 1
                else:
                    enti_dict[tag] += 1
            if 'S-' in tag:  #当S-出现，直接加一
                flag = 0
                tag = tag[2:]
                if tag not in enti_dict:
                    enti_dict[tag] = 1
                else:
                    enti_dict[tag] += 1
            if tag == 'O': #出现O-时，将状态设为0 B_word设为0
                flag = 0
        return enti_dict


    def count_correct_entity(self):
        """
        计算每种实体被正确预测的个数
        address、book、company、game、government、movie、name、organization、position、scene
        :return:
        """
        correct_enti_dict = {}
        flag = 0  #初始状态，表示等待下一个开始状态
        for std_tag, predict_tag in zip(self.std_tags, self.predict_tags):  #zip()用于将数据打包成元组
            if 'B-' in std_tag and std_tag == predict_tag and flag == 0:   #当以B-开头且标签相等时
                flag = 1  #表示已经有B
            if 'M-' in std_tag and std_tag == predict_tag and flag == 1:  #M前已经有过B
                flag = 1
            if 'E-' in std_tag and std_tag == predict_tag and flag == 1:
                flag = 0  #状态重新调整回初始值
                std_tag = std_tag[2:]
                if std_tag not in correct_enti_dict:
                    correct_enti_dict[std_tag] = 1
                else:
                    correct_enti_dict[std_tag] += 1
            if 'S-' in std_tag and std_tag == predict_tag and flag == 0:
                std_tag = std_tag[2:]
                if std_tag not in correct_enti_dict:
                    correct_enti_dict[std_tag] = 1
                else:
                    correct_enti_dict[std_tag] += 1
            if std_tag != predict_tag:
                flag = 0
        return correct_enti_dict


    def count_entity(self, tag_list):
        """
        计算标准列表中的实体个数，因为标准结果中无错误分类，所以实体的个数可以直接计算为E标签和S标签数目之和
        :return:
        """
        entity_count = 0   #记录实体数量
        for tag in tag_list:    #遍历所有标签
            if 'E-' in tag:
                entity_count += 1
            if 'S-' in tag:
                entity_count += 1
        return entity_count

def flatten_lists(lists):
    """
    将列表的列表拼成一个列表
    :param lists:
    :return:
    """
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list
