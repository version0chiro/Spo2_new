def stringGetValue(test,num):
    return test.replace('{','').replace('}','').replace('\n','').split(',')[num].split(' ')[-1].split(':')[-1]