def stringGetValue(test,num):
    return float(test.replace('{','').replace('}','').replace('\n','').split(',')[num].split(':')[-1])
    # return test.replace('{','').replace('}','').replace('\n','').split(',')[num].split(' ')[-1].split(':')[-1]