import ttcanal


def __main__():
    # fName = input("Enter name of file to be analyzed. \n >>")
    fName = "./constant.TXT"
    arr1 = ttcanal.ttc(fName)
    print (arr1.frames)
    arr1.circleSeek()
    return


__main__()
