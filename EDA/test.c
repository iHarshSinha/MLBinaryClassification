#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int data;
    struct node *next;
} Node;

Node* new_node(int x) {
    Node *t = malloc(sizeof(Node));
    t->data = x;
    t->next = NULL;
    return t;
}

Node* insert_front(Node *head, int x) {
    Node *t = new_node(x);
    t->next = head;       // link
    return t;
}

Node* insert_end(Node *head, int x) {
    Node *t = new_node(x);
    if (head == NULL) return t;

    Node *cur = head;
    while (cur->next != NULL)    // go last
        cur = cur->next;

    cur->next = t;               // attach
    return head;
}

Node* insert_pos(Node *head, int x, int pos) {
    if (pos <= 0) return insert_front(head, x);

    Node *cur = head;
    int i = 0;

    while (cur != NULL && i < pos - 1) { // move
        cur = cur->next;
        i++;
    }

    if (cur == NULL) return insert_end(head, x);

    Node *t = new_node(x);
    t->next = cur->next;  // link
    cur->next = t;
    return head;
}

Node* insert_front_rec(Node *head, int x) {
    Node *t = new_node(x);
    t->next = head;
    return t;
}

Node* insert_end_rec(Node *head, int x) {
    if (head == NULL) return new_node(x);
    head->next = insert_end_rec(head->next, x); // go next
    return head;
}

Node* insert_pos_rec(Node *head, int x, int pos) {
    if (pos == 0) return insert_front_rec(head, x);
    if (head == NULL) return new_node(x);

    head->next = insert_pos_rec(head->next, x, pos - 1); // decrease
    return head;
}

Node* delete_front(Node *head) {
    if (head == NULL) return NULL;
    Node *t = head->next;   // next
    free(head);             // free
    return t;
}

Node* delete_end(Node *head) {
    if (head == NULL) return NULL;
    if (head->next == NULL) {
        free(head);
        return NULL;
    }

    Node *cur = head;
    while (cur->next->next != NULL) // prev-last
        cur = cur->next;

    free(cur->next);        // delete
    cur->next = NULL;
    return head;
}

Node* delete_pos(Node *head, int pos) {
    if (pos <= 0) return delete_front(head);
    if (head == NULL) return NULL;

    Node *cur = head;
    int i = 0;

    while (cur != NULL && i < pos - 1) {
        cur = cur->next;    // move
        i++;
    }

    if (cur == NULL || cur->next == NULL) return delete_end(head);

    Node *t = cur->next;    // target
    cur->next = t->next;    // unlink
    free(t);
    return head;
}

Node* delete_front_rec(Node *head) {
    if (head == NULL) return NULL;
    Node *t = head->next;
    free(head);
    return t;
}

Node* delete_end_rec(Node *head) {
    if (head == NULL) return NULL;
    if (head->next == NULL) {
        free(head);
        return NULL;
    }
    head->next = delete_end_rec(head->next); // go next
    return head;
}

Node* delete_pos_rec(Node *head, int pos) {
    if (head == NULL) return NULL;
    if (pos == 0) return delete_front_rec(head);

    head->next = delete_pos_rec(head->next, pos - 1); // decrease
    return head;
}

int search(Node *head, int key) {
    Node *cur = head;
    while (cur != NULL) {     // scan
        if (cur->data == key) return 1;
        cur = cur->next;
    }
    return 0;
}

int search_rec(Node *head, int key) {
    if (head == NULL) return 0;
    if (head->data == key) return 1;
    return search_rec(head->next, key);
}

void print_list(Node *head) {
    Node *cur = head;
    while (cur != NULL) {
        printf("%d ", cur->data);
        cur = cur->next;
    }
    printf("\n");
}

void print_list_rec(Node *head) {
    if (head == NULL) { printf("\n"); return; }
    printf("%d ", head->data);
    print_list_rec(head->next);
}

Node* free_list(Node *head) {
    Node *cur = head;
    while (cur != NULL) {
        Node *t = cur->next;  // save
        free(cur);            // free
        cur = t;              // move
    }
    return NULL;
}

Node* free_list_rec(Node *head) {
    if (head == NULL) return NULL;
    free_list_rec(head->next);
    free(head);
    return NULL;
}

int main() {
    Node *head = NULL;

    head = insert_front(head, 2);
    head = insert_front(head, 1);
    head = insert_end(head, 4);
    head = insert_pos(head, 3, 2);
    print_list(head);

    head = insert_front_rec(head, 0);
    head = insert_end_rec(head, 5);
    head = insert_pos_rec(head, 9, 2);
    print_list(head);

    printf("Search 4: %d\n", search(head, 4));
    printf("Search 10: %d\n", search_rec(head, 10));

    head = delete_front(head);
    head = delete_end(head);
    head = delete_pos(head, 1);
    print_list(head);

    head = delete_front_rec(head);
    head = delete_end_rec(head);
    head = delete_pos_rec(head, 1);
    print_list_rec(head);

    head = free_list_rec(head);
    return 0;
}

int isPrime(int x){
    if(x<=1){
        return 0;
    }
    if(x==2){
        return 1;
    }
    for(int i=2;i<n;i++){
        if(x%i==0){
            return 0;
        }
    }
    return 1;
}

int a[]={1,2,3,4}

x1=x2



strcpy(a,b)
a=b