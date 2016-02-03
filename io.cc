/** \file
    Testing reading and writing of nD volumes to various file types.
    \todo append test
    \todo Write tests will fail if rgb is loaded because tiff reader loads
          colors to last dim, but ffmpeg writer assumes color is first dim.
          Need dimension annotation and transpose.
    \todo Add better ndioReadSubarray tests
          1. Read lines from a file
          2. Read images from a stack
          3. Skip planes/channels
          4. Read cropped
          5. Content comparison tests.
          6. Add subarray tests to ndio-series-tests
    @cond TEST
*/

#include <gtest/gtest.h>
#include "config.h"
#include "helpers.h"
#include "nd.h"

#define countof(e) (sizeof(e)/sizeof(*e))

static
struct _files_t
{ const char  *path;
  nd_type_id_t type;
  size_t       ndim;
  size_t       shape[5];
}
 
file_table[] =
{
  {ND_TEST_DATA_PATH"/vol.1ch.tif",nd_i16,3,{620,512,100,1,1}},
  {ND_TEST_DATA_PATH"/vol.rgb.tif",nd_u8 ,4,{620,512, 39,3,1}},
  {ND_TEST_DATA_PATH"/vol.rgb.mp4",nd_u8 ,4,{620,512, 39,3,1}},
  {ND_TEST_DATA_PATH"/vol.rgb.ogg",nd_u8 ,4,{620,512, 39,3,1}}, // don't know how to decode properly, strange pts's, jumps from frame 0 to frame 12
  {ND_TEST_DATA_PATH"/vol.rgb.avi",nd_u8 ,4,{620,512, 39,3,1}},
//{ND_TEST_DATA_PATH"/38B06.5-8.lsm",nd_u16,4,{1024,1024,248,4,1}}, // lsm's fail right now bc of the thumbnails
  {0}
};

struct ndio: public testing::Test
{ void SetUp()
  { ndioAddPluginPath(NDIO_BUILD_ROOT);
  }
};

TEST_F(ndio,CloseNULL) { ndioClose(NULL); }

TEST_F(ndio,OpenClose)
{ struct _files_t *cur;
  // Examples that should fail to open
  EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious",NULL,"r"));
  //EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious",NULL,"w"));
  EXPECT_EQ(NULL,ndioOpen("",NULL,"r"));
  EXPECT_EQ(NULL,ndioOpen("",NULL,"w"));
  EXPECT_EQ(NULL,ndioOpen(NULL,NULL,"r"));
  EXPECT_EQ(NULL,ndioOpen(NULL,NULL,"w"));
  // Examples that should open
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    ndioClose(file);
  }
}

TEST_F(ndio,Name)
{ struct _files_t *cur;
  EXPECT_STREQ("(error)",ndioFormatName(NULL));
  // Examples that should open
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    const char* n;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    EXPECT_NE("(error)",n=ndioFormatName(file));
    printf("%s\n",n);
    ndioClose(file);
  }
}

TEST_F(ndio,Get)
{ ndio_t file;
  malloc(1024);
  EXPECT_NE((void*)NULL,file=ndioOpen(file_table[0].path,"tiff/mylib","r"));
  // Get not supported for first file format
  EXPECT_EQ((void*)NULL,ndioGet(file));
  ndioClose(file);
}

TEST_F(ndio,Set)
{ char param[] = {1,2,3,4};
  size_t nbytes = sizeof(param);
  ndio_t file;
  EXPECT_NE((void*)NULL,file=ndioOpen(file_table[0].path,"tiff/mylib","r"));
  // Set not supported for first file format
  EXPECT_EQ((void*)NULL,ndioSet(file,(void*)param,nbytes));
  ndioClose(file);
}

TEST_F(ndio,Shape)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t form;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"))<<cur->path;
    ASSERT_NE((void*)NULL,form=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    EXPECT_EQ(cur->type,ndtype(form))<<cur->path;
    EXPECT_EQ(cur->ndim,ndndim(form))<<cur->path;
    for(size_t i=0;i<cur->ndim;++i)
      EXPECT_EQ(cur->shape[i],ndshape(form)[i])<<cur->path;
    ndfree(form);
    ndioClose(file);
  }
}

TEST_F(ndio,Read)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t vol;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    ASSERT_NE((void*)NULL, vol=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    EXPECT_EQ(vol,ndref(vol,malloc(ndnbytes(vol)),nd_heap));
    EXPECT_EQ(file,ndioRead(file,vol)); // could chain ndioClose(ndioRead(ndioOpen("file.name","r"),vol));
    ndfree(vol);
    ndioClose(file);
  }
}

TEST_F(ndio,ReadSubarray)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t vol;
    size_t n;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    ASSERT_NE((void*)NULL, vol=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    // Assume we know the dimensionality of our data and which dimension to iterate over.
    n=ndshape(vol)[2];      // remember the range over which to iterate
    ndShapeSet(vol,2,1); // prep to iterate over 3'rd dimension (e.g. expect WxHxDxC data, read WxHx1XC planes)
    EXPECT_EQ(vol,ndref(vol,malloc(ndnbytes(vol)),nd_heap)); // alloc just enough data      
    { size_t pos[]={0,0,0,0}; // 4d data
      ndio_t a=file; //temp variable used to terminate loop early if something goes wrong
      for(size_t i=0;i<n && a;++i,++pos[2])
      { ASSERT_EQ(file,a=ndioReadSubarray(file,vol,pos,0))<<ndioError(file); // seek to pos and read, shape limited by vol
      }
    }
    ndfree(vol);
    ndioClose(file);
  }
}

TEST_F(ndio,MethodChainingErrors)
{ nd_t a=0;
  ndioClose(ndioRead (ndioOpen("does.not.exist",NULL,"r"),a));
  ndioClose(ndioWrite(ndioOpen("does.not.exist",NULL,"w"),a));
}

class Write:public ::testing::Test
{ 
public:
  nd_t a;
  ndio_t file;
  Write() :a(0),file(0){}
  void SetUp()
  { ndio_t infile=0;
    ASSERT_NE((void*)NULL,infile=ndioOpen(file_table[0].path,NULL,"r"));
    ASSERT_NE((void*)NULL, a=ndioShape(infile))<<ndioError(infile)<<"\n\t"<<file_table[0].path;
    EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
    ASSERT_EQ(infile,ndioRead(infile,a));
    ndioClose(infile);
  }
  void TearDown()
  { ndfree(a);
  }
};

#define WriteTestInstance(ext) \
  TEST_F(Write,ext) \
  { nd_t vol; \
    ndio_t fin; \
    EXPECT_NE((void*)NULL,ndioWrite(file=ndioOpen("testout."#ext,NULL,"w"),a)); \
    ndioClose(file); \
    EXPECT_NE((void*)NULL,fin=ndioOpen("testout."#ext,NULL,"r")); \
    ASSERT_NE((void*)NULL, vol=ndioShape(fin))<<ndioError(fin)<<"\n\t"<<"testout."#ext; \
    ndioClose(fin); \
    { int i; \
      EXPECT_EQ(-1,i=firstdiff(ndndim(a),ndshape(a),ndshape(vol)))\
          << "\torig shape["<<i<<"]: "<< ndshape(a)[i] << "\n"  \
          << "\tread shape["<<i<<"]: "<< ndshape(vol)[i] << "\n"; \
    } \
  }
WriteTestInstance(tif);
WriteTestInstance(mp4);
WriteTestInstance(m4v);
WriteTestInstance(ogg);
WriteTestInstance(webm);
WriteTestInstance(mov);
WriteTestInstance(h5);
/// @endcond
