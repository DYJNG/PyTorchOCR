CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)

DEPS = $(shell find include -xtype f)
CXX_SOURCES = src/db_postprocess.cpp src/clipper.cpp
OPENCV = `pkg-config --cflags --libs opencv`

LIB_SO = db_postprocess.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)
