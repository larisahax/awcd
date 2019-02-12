LIB_DIR = lib

default: pysparsematrixdot

pysparsematrixdot: setup.py pysparsematrixdot.pyx $(LIB_DIR)/libsparsematrixdot.a
	 rm -f pysparsematrixdot.c && rm -Rf build && python3 setup.py build_ext --inplace

$(LIB_DIR)/libsparsematrixdot.a:
	make -C $(LIB_DIR) libsparsematrixdot.a

clean:
	rm *.so
