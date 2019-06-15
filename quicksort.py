# quick sort
print('Tell me what you wanner to sort:')           # (please use space to split them)

def split(num, l, r):
    i = 0
    j = r
    x = num[i]
    while i <= j:
        while num[j] > x:
            j = j-1
        if j <= i:
            num[i] = x
            return i
        num[i] = num[j]
        i = i + 1
        while num[i] < x:
            i = i + 1
        if j <= i:
            num[j] = x
            i = j
            return i
        num[j] = num[i]
        j = j - 1
def quicksort(num, l, r):
    if (l >= r):
        return
    if (l < 0):
        return
    i = split(num, l, r)
    quicksort(num, l, i)
    quicksort(num, i+1, r)

# loop this program until you don't want to continue
while 1:
    arr = input("")                                 # memory your inputs
    num = [int(n) for n in arr.split()]             # format your inputs to list
    length = len(num)                               # memory your list's length
    quicksort(num, 0, length-1)
    print(num)                                      # print sorted num
    print('please input \'y\' if you\'d like continue, or input otherwise character to stop')
    s = input("")
    if s != 'y':
        break
