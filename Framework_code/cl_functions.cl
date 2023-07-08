#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

bool isRangeSet(int tag){return (tag&16);}
void setTrim1(__global int tags[],int index) { tags[index]|=32;/*atomic_or( &tags[index] ,32);*/ };
bool isTrim1(int tags) { return ( tags & 32); }
void setTrim2(__global int tags[],int index) { atomic_or(&tags[index] ,64); };
bool isTrim2(int tags) { return ( tags & 64); };
bool isForwardVisited(int tags){ return (tags & 1);}
bool isForwardPropagate(int tags){ return (tags & 4);}
void setForwardPropagateBit(__global int tags[],int index)
{ atomic_or(&tags[index],4); };
void clearForwardPropagateBit(__global int tags[],int index){tags[index] = ( tags[index] ^ 4); };
void rangeSet(__global int tags[],int index) { tags[index] = ( tags[index] | 16); };
bool isBackwardVisited(int tags) { return (tags & 2); }
bool isBackwardPropagate(int tags) { return ( tags & 8); }
void setBackwardPropagateBit(__global int tags[],int index) { atomic_or(&tags[index] , 8); };
void clearBackwardPropagateBit(__global int tags[],int index){ tags[index] = ( tags[index] ^ 8);};
bool isForwardProcessed(int tags){return (tags&256);};
void setForwardProcessed(__global int tags[],int index){atomic_or(&tags[index],256);}
void clearForwardProcessed(__global int tags[],int index){ tags[index] = (tags[index] ^ 256);};
bool isBackwardProcessed(int tags){return (tags&512);};
void setBackwardProcessed(__global int tags[],int index){atomic_or(&tags[index],512);}
void clearBackwardProcessed(__global int tags[],int index){ tags[index] = (tags[index] ^ 512);};


void setPivot(__global int tags[],int index) { tags[index] = ( tags[index] | 128); };
bool isPivot(int tags) { return ( tags & 128); }




__kernel void trim1(__global int *tags,
                    __global int *Fc,
                    __global int *Fr,
                    __global int *Bc,
                    __global int *Br,
                    __global unsigned *colors,
                    __global bool *terminate,
                    __global int *trim1_cnt )
{

   
    int vertex=get_global_id(0);

    //  for(;vertex<=21297772;vertex+=1){
        
        if(vertex>0 )
        {
            if(isTrim1(tags[vertex]))
            return;//continue;

        //  printf("Hello,vertex=%d\n",vertex);
            
            bool eliminate=true;
            int starting_index=Br[vertex];
            int ending_index=Br[vertex+1];

            
        //  printf("vertex=%d,starting_index=%d,end_index=%d\n",
            //                       vertex,starting_index,ending_index);
                

            for(int i=starting_index;i<ending_index;i++)
            {
                int neighbour_vertex=Bc[i];
                
                if(!isTrim1(tags[neighbour_vertex]) && colors[vertex]==colors[neighbour_vertex])
                {
                    eliminate=false;
                    break;
                }
            }
        
            if(!eliminate)
            {
                eliminate=true;
                starting_index=Fr[vertex];
                ending_index=Fr[vertex+1];

             
                for(int i=starting_index;i<ending_index;i++)
                {
                    int neighbour_vertex=Fc[i];
                   
                    // if(neighbour_vertex<0)
                    // {
                    //     printf("This is the culprit1,vertex=%d,i=%d\n",vertex,i);
                    // }
                    // else if(neighbour_vertex>3774768)
                    // {
                    //     printf("This is the culprit2,vertex=%d,i=%d\n",vertex,i);
                    // }
                    if(!isTrim1(tags[neighbour_vertex]) && colors[vertex]==colors[neighbour_vertex])
                    {
                        eliminate=false;
                        break;
                    }
                }
            }
            
            if(eliminate)
            {
               // setTrim1(tags,vertex);
                setTrim1(tags,vertex);
                *terminate=false;
                atomic_add(trim1_cnt,1);
              //  printf("%d\n",vertex);
            // printf("tags:%d,vertex:%d\n",tags[vertex],vertex);
            }
            
        }
  
   //  }

}
