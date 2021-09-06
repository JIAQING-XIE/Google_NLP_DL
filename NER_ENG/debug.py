def debug_1(group1, group2):
    new_group = []
    i = 0
    j = 0
    while i < len(group1):
        while j < len(group2):
            if group1[i] == group2[j]:
                new_group.append(group1[i])
                i+=1 # move to next
                j+=1 # move to next
            elif group1[i] in group2[j]: # meet mark behind
                current = i
                for current in range(i, len(group1) + 1):
                    if ''.join(group1[i:current]) in group2[j]:
                        if ''.join(group1[i:current]) == group2[j]:
                            new_group.append(group2[j])
                            i = current
                            j+=1
                            break
                    else:
                        break  
    return new_group

if __name__ == "__main__":
    group1 = ["1", "-", "1", "-", "1", "group", "has", "a", "lot", "-", "common", "-", "dog"]
    group2 = ["1-1-1", "group", "has", "a", "lot-common-", "dog"]
    ans = debug_1(group1, group2)
    print("group1:{}".format(group1))
    print("group2:{}".format(group2))
    print(ans)