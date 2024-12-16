#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mkl.h>

#define  NUM_THREADS (1)
#define  BATCH_SIZE  ((128)) // # of sentences
#define  NUM_STEPS   (10)    // # words in each sentence
#define  NUM_DIM     (1024)  // vector representation of a word
#define  NUM_UNITS   ((1024))  // # of hidden LSTM units

int main(int argc, char **argv)
{
  float *w_fh, *w_fx;
  float *w_ih, *w_ix;
  float *w_oh, *w_ox;
  float *w_ch, *w_cx;

  float *x_t, *h_t;
  float *b_f, *b_i, *b_o, *b_c;

  float *w_concat, *hx_concat, *b_concat;

  float *res, *c_state, *history;
  float tmp;
  int i, j, t;

  double total_time, t_start, t_end;
  double gemm_time=0.0, gemm_start, gemm_end;
  double sigmoid_time=0.0, sigmoid_start, sigmoid_end;
  double tanh_time=0.0, tanh_start, tanh_end;
  double bias_time=0.0, bias_start, bias_end;
  double elementwise_time=0.0, elementwise_start, elementwise_end;

  w_fh = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_UNITS, 128);
  w_ih = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_UNITS, 128);
  w_oh = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_UNITS, 128);
  w_ch = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_UNITS, 128);

  w_fx = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_DIM, 128);
  w_ix = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_DIM, 128);
  w_ox = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_DIM, 128);
  w_cx = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*NUM_DIM, 128);

  h_t  = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*BATCH_SIZE, 128);
  x_t  = (float *)mkl_malloc(sizeof(float)*NUM_DIM*BATCH_SIZE, 128);
  c_state = (float *)mkl_malloc(sizeof(float)*NUM_STEPS*BATCH_SIZE*NUM_UNITS, 128);
  history = (float *)mkl_malloc(sizeof(float)*NUM_STEPS*BATCH_SIZE*NUM_UNITS, 128);

  b_f  = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*BATCH_SIZE, 128);
  b_i  = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*BATCH_SIZE, 128);
  b_o  = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*BATCH_SIZE, 128);
  b_c  = (float *)mkl_malloc(sizeof(float)*NUM_UNITS*BATCH_SIZE, 128);

  w_concat  = (float *)mkl_malloc(sizeof(float)*(NUM_UNITS*4)*(NUM_UNITS+NUM_DIM), 128);
  hx_concat = (float *)mkl_malloc(sizeof(float)*(NUM_UNITS+NUM_DIM)*BATCH_SIZE, 128);
  b_concat  = (float *)mkl_malloc(sizeof(float)*(NUM_UNITS*4)*BATCH_SIZE, 128);
  res       = (float *)mkl_malloc(sizeof(float)*(NUM_UNITS*4)*BATCH_SIZE, 128);

  // Random initialize data arrays
  for (i=0; i<NUM_UNITS; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      w_fh[i*NUM_UNITS+j] = w_ih[i*NUM_UNITS+j] = w_oh[i*NUM_UNITS+j] = w_ch[i*NUM_UNITS+j] =  rand()/(RAND_MAX - 0.5);
    }
  }

  for (i=0; i<NUM_DIM; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      w_fx[i*NUM_UNITS+j] = w_ix[i*NUM_UNITS+j] = w_ox[i*NUM_UNITS+j] = w_cx[i*NUM_UNITS+j] = rand()/(RAND_MAX - 0.5);
    }
  }

  for (i=0; i<BATCH_SIZE; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      h_t[i*NUM_UNITS+j] = rand()/(RAND_MAX - 0.5);
    }
  }

  for (i=0; i<BATCH_SIZE; i++) {
    for (j=0; j<NUM_DIM; j++) {
      x_t[i*NUM_DIM+j] = rand()/(RAND_MAX - 0.5);
    }
  }

  for (i=0; i<BATCH_SIZE; i++) {
    tmp = rand()/(RAND_MAX - 0.5);
    for (j=0; j<NUM_UNITS; j++) {
      b_f[i*NUM_UNITS+j] = b_i[i*NUM_UNITS+j] = b_o[i*NUM_UNITS+j] = b_c[i*NUM_UNITS+j] = tmp;
    }
  }

  // Initialize cell state at t=0 to zero
  for (i=0; i<BATCH_SIZE; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      c_state[i*NUM_UNITS+j] = 0.0;
    }
  }

  /*
   * ___________________
     |        |        |
     |  W_fh  |  W_fx  |
     |________|________|
     |        |        |
     |  W_ih  |  W_ix  |
     |________|________|
     |        |        |
     |  W_oh  |  W_ox  |
     |________|________|
     |        |        |
     |  W_ch  |  W_cx  |
     |________|________|
   *
   */

  int ld = NUM_UNITS*4;
  for (i=0; i<NUM_UNITS; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      w_concat[i*ld+j]             = w_fh[i*NUM_UNITS+j];
      w_concat[i*ld+j+NUM_UNITS]   = w_ih[i*NUM_UNITS+j];
      w_concat[i*ld+j+2*NUM_UNITS] = w_oh[i*NUM_UNITS+j];
      w_concat[i*ld+j+3*NUM_UNITS] = w_ch[i*NUM_UNITS+j];
    }
  }

  ld = NUM_UNITS*4;
  for (i=0; i<NUM_DIM; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      w_concat[(i+NUM_UNITS)*ld+j]             = w_fx[i*NUM_UNITS+j];
      w_concat[(i+NUM_UNITS)*ld+j+NUM_UNITS]   = w_ix[i*NUM_UNITS+j];
      w_concat[(i+NUM_UNITS)*ld+j+2*NUM_UNITS] = w_ox[i*NUM_UNITS+j];
      w_concat[(i+NUM_UNITS)*ld+j+3*NUM_UNITS] = w_cx[i*NUM_UNITS+j];
    }
  }

  /*
   * _________
     |       |
     |  h_t  |
     |_______|
     |       |
     |  x_t  |
     |_______|
   *
   */

  ld = NUM_UNITS+NUM_DIM;
  for (i=0; i<BATCH_SIZE; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      hx_concat[i*ld+j] = h_t[i*NUM_UNITS+j];
    }
  }

  for (i=0; i<BATCH_SIZE; i++) {
    for (j=0; j<NUM_DIM; j++) {
      hx_concat[i*ld+j+NUM_UNITS] = x_t[i*NUM_DIM+j];
    }
  }
  /*
   * _________
     |       |
     |  b_f  |
     |_______|
     |       |
     |  b_i  |
     |_______|
     |       |
     |  b_o  |
     |_______|
     |       |
     |  b_c  |
     |_______|
   *
   */


  ld = NUM_UNITS*4;
  for (i=0; i<BATCH_SIZE; i++) {
    for (j=0; j<NUM_UNITS; j++) {
      b_concat[i*ld+j]             = b_f[i*NUM_UNITS+j];
      b_concat[i*ld+j+NUM_UNITS]   = b_i[i*NUM_UNITS+j];
      b_concat[i*ld+j+2*NUM_UNITS] = b_o[i*NUM_UNITS+j];
      b_concat[i*ld+j+3*NUM_UNITS] = b_c[i*NUM_UNITS+j];
    }
  }

  int m = NUM_UNITS*4;
  int k = NUM_UNITS+NUM_DIM;
  int n = BATCH_SIZE;
  int lda = m;
  int ldb = k;
  int ldc = m;
  float alpha = 1.0;
  float beta  = 1.0;

  dsecnd();
  dsecnd();
  dsecnd();
  // Warm-up GEMM
  sgemm("N", "N", &m, &n, &k, &alpha, w_concat, &lda, hx_concat, &ldb, &beta, res, &ldc);

  t_start = dsecnd();
  for (t=1; t<NUM_STEPS; t++) {
  // Initialize the result matrix to bias values
  /*          
   * _________     _________
     |       |     |       |
     |   f   |  =  |  b_f  |
     |_______|     |_______|
     |       |     |       |
     |   i   |  =  |  b_i  |
     |_______|     |_______|
     |       |     |       |
     |   o   |  =  |  b_o  |
     |_______|     |_______|
     |       |     |       |
     |   c   |  =  |  b_c  |
     |_______|     |_______|
   *
   */
    bias_start = dsecnd();
    for (i=0; i<BATCH_SIZE; i++) {
      for (j=0; j<NUM_UNITS*4; j++) {
        res[i*NUM_UNITS*4+j] = b_concat[i*NUM_UNITS*4+j];
      }
    }
    bias_end = dsecnd();
    bias_time += bias_end - bias_start;

    // Do MatMul
  /*
   * ___________________   __________     __________
     |        |        |   |        |     |        |
     |  W_fh  |  W_fx  |   |   h_t' |     |   f    |
     |________|________|   |________|     |________|
     |        |        |   |        |     |        |
     |  W_ih  |  W_ix  |   |   x_t  |     |   i    |
     |________|________| * |________|  =  |________|
     |        |        |                  |        |
     |  W_oh  |  W_ox  |                  |   o    |
     |________|________|                  |________|
     |        |        |                  |        |
     |  W_ch  |  W_cx  |                  |   c    |
     |________|________|                  |________|
   *
   */
    gemm_start = dsecnd();
    sgemm("N", "N", &m, &n, &k, &alpha, w_concat, &lda, hx_concat, &ldb, &beta, res, &ldc);
    gemm_end = dsecnd();
    gemm_time += gemm_end - gemm_start;

    // Do sigmoid on f, i, o
    sigmoid_start = dsecnd();
    for (i=0; i<BATCH_SIZE; i++) {
      for (j=0; j<NUM_UNITS*3; j++) {
        tmp = res[i*NUM_UNITS*4+j];
        res[i*NUM_UNITS*4+j] = 1/(1+expf(-tmp));
      }
    }
    sigmoid_end = dsecnd();
    sigmoid_time += sigmoid_end - sigmoid_start;

    // Do tanh on c (cell state)
    tanh_start = dsecnd();
    for (i=0; i<BATCH_SIZE; i++) {
      for (j=0; j<NUM_UNITS; j++) {
        res[i*NUM_UNITS*4+j+NUM_UNITS*3] = tanhf(res[i*NUM_UNITS*4+j+NUM_UNITS*3]);
      }
    }
    tanh_end = dsecnd();
    tanh_time += tanh_end - tanh_start;

    // Get new cell state, c_t = f*c(t-1) + i*c; *=Elementwise-mul
    elementwise_start = dsecnd();
    for (i=0; i<BATCH_SIZE; i++) {
      for (j=0; j<NUM_UNITS; j++) {
        c_state[t*(BATCH_SIZE*NUM_UNITS)+i*NUM_UNITS+j] = res[i*4*NUM_UNITS+j] * c_state[(t-1)*(BATCH_SIZE*NUM_UNITS)+i*NUM_UNITS+j] + res[i*4*NUM_UNITS+j+NUM_UNITS] * res[i*4*NUM_UNITS+j+3*NUM_UNITS];
      }
    }
    elementwise_end = dsecnd();
    elementwise_time += elementwise_end - elementwise_start;

    // Tanh on new cell state, c_t
    tanh_start = dsecnd();
    for (i=0; i<BATCH_SIZE; i++) {
      for (j=0; j<NUM_UNITS; j++) {
        c_state[t*(BATCH_SIZE*NUM_UNITS)+i*NUM_UNITS+j] = tanhf(c_state[t*(BATCH_SIZE*NUM_UNITS)+i*NUM_UNITS+j]);
      }
    }
    tanh_end = dsecnd();
    tanh_time += tanh_end - tanh_start;

    // multiply new cell state with o to get h_t, and update hx_concat
    elementwise_start = dsecnd();
    ld = NUM_UNITS+NUM_DIM;
    for (i=0; i<BATCH_SIZE; i++) {
      for (j=0; j<NUM_UNITS; j++) {
        hx_concat[i*ld+j] =  res[(i*4*NUM_UNITS)+j+2*NUM_UNITS] * c_state[t*(BATCH_SIZE*NUM_UNITS)+i*NUM_UNITS+j];
        history[t*(BATCH_SIZE*NUM_UNITS)+i*NUM_UNITS+j] = hx_concat[i*ld+j];
      }
    }
    elementwise_end = dsecnd();
    elementwise_time += elementwise_end - elementwise_start;
  }
  t_end = dsecnd();
  total_time = t_end - t_start;

  printf ("\n Total time = %.2f\n GEMM time = %.2f (%.2f)\n Sigmoid time = %.2f (%.2f)\n Tanh time = %.2f (%.2f)\n Bias-ops time = %.2f (%.2f)\n Elementwise-ops time = %.2f (%.2f)\n",
          (total_time)*1.0e3,
          gemm_time*1.0e3, (gemm_time/total_time)*100.,
          sigmoid_time*1.0e3, (sigmoid_time/total_time)*100.,
          tanh_time*1.e03, (tanh_time/total_time)*100.,
          bias_time*1.e03, (bias_time/total_time)*100.,
          elementwise_time*1.e03, (elementwise_time/total_time)*100.);

  mkl_free(w_fh);
  mkl_free(w_fx);
  mkl_free(w_ih);
  mkl_free(w_ix);
  mkl_free(w_oh);
  mkl_free(w_ox);
  mkl_free(w_ch);
  mkl_free(w_cx);
  mkl_free(x_t);
  mkl_free(h_t);
  mkl_free(b_f);
  mkl_free(b_i);
  mkl_free(b_o);
  mkl_free(b_c);
  mkl_free(w_concat);
  mkl_free(hx_concat);
  mkl_free(b_concat);
  mkl_free(res);
  mkl_free(c_state);
  mkl_free(history);

  return 0;
}

