import os


if __name__ == "__main__":
    input = os.path.expanduser("~/.cache/financial_risk_assessment/FIN5.txt")
    output = os.path.expanduser("~/.cache/financial_risk_assessment/train.txt")
    lines = []
    count = 0
    with open(input,'r',encoding='utf-8') as f:
        pre_tag = None

        for line in f:
            line = line[:-1]
            tokens = line.split()
            if len(tokens) == 0:
                lines.append(line)
                continue
            tag = tokens[-1]
            if len(tag) > 1 and pre_tag == 'O':
                new_tag = 'B'+tag[1:]
                tokens[-1] = new_tag

            pre_tag = tag
            lines.append(" ".join(tokens))
            # count += 1
            # if count > 10:
            #     break

    with open(output,'w',encoding='utf-8') as f:
        for line in lines:
            f.write(line+'\n')



