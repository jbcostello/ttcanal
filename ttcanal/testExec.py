import ttcanal


def __main__():
    # fName = input("Enter name of file to be analyzed. \n >>")
    fName = "./5-29_calib_01.TXT"
    arr1 = ttcanal.ttc(fName)
    arr1.getframes()
    arr1.circleSeek()
    '''
    arr1.animate(title='animateTest')
    arr1.sdmap(title='sdTest', Plot=False)
    arr1.sdavgmap(title='sdavgTest', Plot=False)
    arr1.avgmap(title='avgTest', Plot=False)
    arr1.changeMap(title='changeTest', Plot=False)
    arr1.rawmap(title='rawTest', Plot=False)
    arr1.pixelRange(5, title='pixelTest', Plot=False)
    arr1.smearMap(5, title='smearTest', Plot=False)
    arr1.cam_calib()
    # arr1.line_cut(title='lineTest')
    arr1.plot_bin(5)
    arr1.fourier2D()
    '''

    return


__main__()
