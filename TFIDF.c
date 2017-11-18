#include <dirent.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORDS_IN_CORPUS 32
#define MAX_FILEPATH_LENGTH 16
#define MAX_WORD_LENGTH 16
#define MAX_DOCUMENT_NAME_LENGTH 8
#define MAX_STRING_LENGTH 64

#define ROOT 0

typedef char word_document_str[MAX_STRING_LENGTH];

typedef struct o {
  char word[32];
  char document[8];
  int wordCount;
  int docSize;
  int numDocs;
  int numDocsWithWord;
} obj;

typedef struct w {
  char word[32];
  int numDocsWithWord;
  int currDoc;
} u_w;

static int myCompare(const void *a, const void *b) { return strcmp(a, b); }

int main(int argc, char *argv[]) {
  DIR *files;
  struct dirent *file;
  int i = 0, j = 0;
  int numDocs = 0, docSize = 0, contains = 0;
  char filename[MAX_FILEPATH_LENGTH], word[MAX_WORD_LENGTH],
      document[MAX_DOCUMENT_NAME_LENGTH];

  printf("%d\n", __LINE__);

  // Will hold all TFIDF objects for all documents
  obj TFIDF[MAX_WORDS_IN_CORPUS];
  int TF_idx = 0;

  // Will hold all unique words in the corpus and the number of documents with
  // that word
  u_w unique_words[MAX_WORDS_IN_CORPUS];
  int uw_idx = 0;

  // Will hold the final strings that will be printed out
  word_document_str strings[MAX_WORDS_IN_CORPUS];

  // Count numDocs
  if ((files = opendir("input")) == NULL) {
    printf("Directory failed to open\n");
    exit(1);
  }
  while ((file = readdir(files)) != NULL) {
    // On linux/Unix we don't want current and parent directories
    if (!strcmp(file->d_name, "."))
      continue;
    if (!strcmp(file->d_name, ".."))
      continue;
    numDocs++;
  }

  printf("Line no : %d \n", __LINE__);
  MPI_Init(&argc, &argv);
  int rank, no_of_processes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &no_of_processes);

  int global_TF_idx = 0, global_uw_idx = 0;
  int no_of_workers = no_of_processes - 1;

  int distribution = numDocs / no_of_workers;

  printf("numDocs -> %d\n", numDocs);

  int *dist_arr = (int *)malloc(sizeof(int) * no_of_processes);

  for (i = 0; i < no_of_processes; i++) {
    if (i == 0)
      dist_arr[i] = 0;
    else
      dist_arr[i] = distribution;
  }

  int remaining_files = numDocs % no_of_workers;
  i = 0;
  while (remaining_files) {

    printf("%d\n", remaining_files);
    if (i == 0)
      dist_arr[i] = 0;
    else {
      dist_arr[i]++;
      remaining_files--;
    }
    printf("%d --> %d\n", dist_arr[i], i);
    i++;
  }

  for (int i = 1; i < no_of_processes; i++) {
    dist_arr[i] = dist_arr[i] + dist_arr[i - 1];
  }

  // print the dist arr
  printf("rank : %d dist_arr :", rank);
  for (i = 0; i < no_of_processes; i++) {
    printf("%d ", dist_arr[i]);
  }
  printf("\n");

  // Loop through each document and gather TFIDF variables for each word
  for (i = 1; i <= numDocs; i++) {

    if (rank == 0)
      break;
    if (!(i > dist_arr[rank - 1] && i <= dist_arr[rank]))
      continue;
    printf("Rank : %d ---->  File : %d \n", rank, i);

    sprintf(document, "doc%d", i);
    sprintf(filename, "input/%s", document);
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
      printf("Error Opening File: %s\n", filename);
      exit(0);
    }

    // Get the document size
    docSize = 0;
    while ((fscanf(fp, "%s", word)) != EOF)
      docSize++;

    // For each word in the document
    fseek(fp, 0, SEEK_SET);
    while ((fscanf(fp, "%s", word)) != EOF) {
      contains = 0;

      // If TFIDF array already contains the word@document, just increment
      // wordCount and break
      for (j = 0; j < TF_idx; j++) {
        if (!strcmp(TFIDF[j].word, word) &&
            !strcmp(TFIDF[j].document, document)) {
          contains = 1;
          TFIDF[j].wordCount++;
          break;
        }
      }

      // If TFIDF array does not contain it, make a new one with wordCount=1
      if (!contains) {
        strcpy(TFIDF[TF_idx].word, word);
        strcpy(TFIDF[TF_idx].document, document);
        TFIDF[TF_idx].wordCount = 1;
        TFIDF[TF_idx].docSize = docSize;
        TFIDF[TF_idx].numDocs = numDocs;
        TF_idx++;
      }

      contains = 0;
      // If unique_words array already contains the word, just increment
      // numDocsWithWord
      for (j = 0; j < uw_idx; j++) {
        if (!strcmp(unique_words[j].word, word)) {
          contains = 1;
          if (unique_words[j].currDoc != i) {
            unique_words[j].numDocsWithWord++;
            unique_words[j].currDoc = i;
          }
          break;
        }
      }

      // If unique_words array does not contain it, make a new one with
      // numDocsWithWord=1
      if (!contains) {
        strcpy(unique_words[uw_idx].word, word);
        unique_words[uw_idx].numDocsWithWord = 1;
        unique_words[uw_idx].currDoc = i;
        uw_idx++;
      }
    }

    fclose(fp);
  }

  MPI_Reduce(&TF_idx, &global_TF_idx, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&uw_idx, &global_uw_idx, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  printf("global_TF_idx : %d   global_uw_index : %d \n", global_TF_idx, global_uw_idx);

  MPI_Request request1, request2;
  if (rank != ROOT) {
    for (int i = 0; i < TF_idx; i++) {
      printf("Sending %d %d ", rank, i);
      MPI_Send(&TFIDF[i], sizeof(obj), MPI_BYTE, 0, 11, MPI_COMM_WORLD);
    }
  }

  for (int i = 0; i < uw_idx; i++) {
    MPI_Send(&unique_words[i], sizeof(u_w), MPI_BYTE, 0, 22, MPI_COMM_WORLD);
  }

  if (rank == ROOT) {

    int k = 0;
    for (int i = 0; i < global_TF_idx; i++) {
        printf("Receiving from %d obj %d \n", i, k);
        MPI_Recv(&TFIDF[k++], sizeof(obj), MPI_BYTE, MPI_ANY_SOURCE, 11, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    int cur_uw_idx = 0;
    for (int i = 0; i < global_uw_idx; i++) {
        contains = 0;
        printf("Receiving from: %d   u_w: %d \n", i, cur_uw_idx);
        u_w temp;
        MPI_Recv(&temp, sizeof(obj), MPI_BYTE, MPI_ANY_SOURCE, 22, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (j = 0; j < cur_uw_idx; j++) {
          if (!strcmp(unique_words[j].word, temp.word)) {
            contains = 1;
            if (unique_words[j].currDoc != temp.currDoc) {
              unique_words[j].numDocsWithWord += temp.numDocsWithWord;
              unique_words[j].currDoc = temp.currDoc;
            }
            break;
          }
        }

        // If unique_words array does not contain it, make a new one with
        // numDocsWithWord=1
        if (!contains) {
          printf("%s  %d %d \n", temp.word, temp.numDocsWithWord,temp.currDoc);
          strcpy(unique_words[cur_uw_idx].word, temp.word);
          unique_words[cur_uw_idx].numDocsWithWord = temp.numDocsWithWord;
          unique_words[cur_uw_idx].currDoc = temp.currDoc;
          cur_uw_idx++;
        }
    }
  }
  
  if (rank == 0) {
    TF_idx = global_TF_idx;
    uw_idx = global_uw_idx;
    // Print TF job similar to HW4/HW5 (For debugging purposes)
    printf("-------------TF Job-------------\n");
    for (j = 0; j < global_TF_idx; j++)
      printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document,
             TFIDF[j].wordCount, TFIDF[j].docSize);

    // Use unique_words array to populate TFIDF objects with: numDocsWithWord
    for (i = 0; i < global_TF_idx; i++) {
      for (j = 0; j < global_uw_idx; j++) {
        printf("%d ",j);
        if (!strcmp(TFIDF[i].word, unique_words[j].word)) {
          TFIDF[i].numDocsWithWord = unique_words[j].numDocsWithWord;
          break;
        }
      }
    }

    // Print IDF job similar to HW4/HW5 (For debugging purposes)
    printf("------------IDF Job-------------\n");
    for (j = 0; j < global_uw_idx; j++)
      printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document,
             TFIDF[j].numDocs, TFIDF[j].numDocsWithWord);

    // Calculates TFIDF value and puts: "document@word\tTFIDF" into strings
    // array
    for (j = 0; j < TF_idx; j++) {
      double TF = 1.0 * TFIDF[j].wordCount / TFIDF[j].docSize;
      double IDF = log(1.0 * TFIDF[j].numDocs / TFIDF[j].numDocsWithWord);
      double TFIDF_value = TF * IDF;
      sprintf(strings[j], "%s@%s\t%.16f", TFIDF[j].document, TFIDF[j].word,
              TFIDF_value);
    }

    // Sort strings and print to file
    qsort(strings, TF_idx, sizeof(char) * MAX_STRING_LENGTH, myCompare);
    FILE *fp = fopen("output.txt", "w");
    if (fp == NULL) {
      printf("Error Opening File: output.txt\n");
      exit(0);
    }
    for (i = 0; i < TF_idx; i++)
      fprintf(fp, "%s\n", strings[i]);
    fclose(fp);
  }
  return 0;
}
