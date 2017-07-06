import atexit

try:
    import bioformats
    import javabridge
except:
    from read_microscopy_image import imread
else:
    from read_bioformats_image import imread

    @atexit.register
    def kill_java():
        print "You are now leaving the Python sector."
        javabridge.kill_vm()