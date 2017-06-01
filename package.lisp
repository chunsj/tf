(cl:defpackage :tf
  (:use #:common-lisp
        #:cffi))

(defparameter *tensorflow-lib* "/usr/local/lib/libtensorflow.dylib")
(defvar *tensorflow-loaded* nil)

(unless *tensorflow-loaded*
  (progn
    (cffi:load-foreign-library *tensorflow-lib*)
    (setf *tensorflow-loaded* T)))

;; // --------------------------------------------------------------------------
;; // C API for TensorFlow.
;; //
;; // The API leans towards simplicity and uniformity instead of convenience
;; // since most usage will be by language specific wrappers.
;; //
;; // Conventions:
;; // * We use the prefix TF_ for everything in the API.
;; // * Objects are always passed around as pointers to opaque structs
;; //   and these structs are allocated/deallocated via the API.
;; // * TF_Status holds error information.  It is an object type
;; //   and therefore is passed around as a pointer to an opaque
;; //   struct as mentioned above.
;; // * Every call that has a TF_Status* argument clears it on success
;; //   and fills it with error info on failure.
;; // * unsigned char is used for booleans (instead of the 'bool' type).
;; //   In C++ bool is a keyword while in C99 bool is a macro defined
;; //   in stdbool.h. It is possible for the two to be inconsistent.
;; //   For example, neither the C99 nor the C++11 standard force a byte
;; //   size on the bool type, so the macro defined in stdbool.h could
;; //   be inconsistent with the bool keyword in C++. Thus, the use
;; //   of stdbool.h is avoided and unsigned char is used instead.
;; // * size_t is used to represent byte sizes of objects that are
;; //   materialized in the address space of the calling process.
;; // * int is used as an index into arrays.
;; //
;; // Questions left to address:
;; // * Might at some point need a way for callers to provide their own Env.
;; // * Maybe add TF_TensorShape that encapsulates dimension info.
;; //
;; // Design decisions made:
;; // * Backing store for tensor memory has an associated deallocation
;; //   function.  This deallocation function will point to client code
;; //   for tensors populated by the client.  So the client can do things
;; //   like shadowing a numpy array.
;; // * We do not provide TF_OK since it is not strictly necessary and we
;; //   are not optimizing for convenience.
;; // * We make assumption that one session has one graph.  This should be
;; //   fine since we have the ability to run sub-graphs.
;; // * We could allow NULL for some arguments (e.g., NULL options arg).
;; //   However since convenience is not a primary goal, we don't do this.
;; // * Devices are not in this API.  Instead, they are created/used internally
;; //   and the API just provides high level controls over the number of
;; //   devices of each type.

;; // --------------------------------------------------------------------------
;; // TF_Version returns a string describing version information of the
;; // TensorFlow library. TensorFlow using semantic versioning.
;; extern size_t TF_DataTypeSize(TF_DataType dt);
(cffi:defcfun ("TF_Version" %TF-VERSION) :string)

;; // --------------------------------------------------------------------------
;; // TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
;; // The enum values here are identical to corresponding values in types.proto.
(cffi:defcenum datatype
  (:float 1)
  (:double 2)
  (:int32 3)
  (:uint8 4)
  (:int16 5)
  (:int8 6)
  (:string 7)
  (:complex64 8)
  (:complex 8)
  (:int64 9)
  (:bool 10)
  (:qint8 11)
  (:quint8 12)
  (:qint32 13)
  (:bfloat16 14)
  (:qint16 15)
  (:quint16 16)
  (:uint16 17)
  (:complex128 18)
  (:half 19)
  (:resource 20))

;; // TF_DataTypeSize returns the sizeof() for the underlying type corresponding
;; // to the given TF_DataType enum value. Returns 0 for variable length types
;; // (eg. TF_STRING) or on failure.
(cffi:defctype size-t :int64)
(cffi:defcfun ("TF_DataTypeSize" %TF-DATATYPESIZE) size-t (dt datatype))

;; // --------------------------------------------------------------------------
;; // TF_Code holds an error code.  The enum values here are identical to
;; // corresponding values in error_codes.proto.
(cffi:defcenum code
  (:ok 0)
  (:cancelled 1)
  (:unknown 2)
  (:invalid-argument 3)
  (:deadline-exceeded 4)
  (:not-found 5)
  (:already-exists 6)
  (:permission-denied 7)
  (:unauthenticated 16)
  (:resource-exhausted 8)
  (:failed-precondition 9)
  (:aborted 10)
  (:out-of-range 11)
  (:unimplemented 12)
  (:internal 13)
  (:unavailable 14)
  (:data-loss 15))

;; // --------------------------------------------------------------------------
;; // TF_Status holds error information.  It either has an OK code, or
;; // else an error code with an associated error message.
(cffi:defctype status :pointer)
;; // Return a new status object.
;; extern TF_Status* TF_NewStatus();
(cffi:defcfun ("TF_NewStatus" %TF-NEWSTATUS) status)
;; // Delete a previously created status object.
;; extern void TF_DeleteStatus(TF_Status*);
(cffi:defcfun ("TF_DeleteStatus" %TF-DELETESTATUS) :void (s status))
;; // Record <code, msg> in *s.  Any previous information is lost.
;; // A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
;; extern void TF_SetStatus(TF_Status* s, TF_Code code, const char* msg);
(cffi:defcfun ("TF_SetStatus" %TF-SETSTATUS) :void (s status) (c code) (m :string))
;; // Return the code record in *s.
;; extern TF_Code TF_GetCode(const TF_Status* s);
(cffi:defcfun ("TF_GetCode" %TF-GETCODE) code (s status))
;; // Return a pointer to the (null-terminated) error message in *s.  The
;; // return value points to memory that is only usable until the next
;; // mutation to *s.  Always returns an empty string if TF_GetCode(s) is
;; // TF_OK.
;; extern const char* TF_Message(const TF_Status* s);
(cffi:defcfun ("TF_Message" %TF-MESSAGE) :string (s status))

;; // --------------------------------------------------------------------------
;; // TF_Buffer holds a pointer to a block of data and its associated length.
;; // Typically, the data consists of a serialized protocol buffer, but other data
;; // may also be held in a buffer.
;; //
;; // By default, TF_Buffer itself does not do any memory management of the
;; // pointed-to block.  If need be, users of this struct should specify how to
;; // deallocate the block by setting the `data_deallocator` function pointer.
;; typedef struct TF_Buffer {
;;   const void* data;
;;   size_t length;
;;   void (*data_deallocator)(void* data, size_t length);
;; } TF_Buffer;
(cffi:defcstruct sbuffer
  (data :pointer)
  (length size-t)
  (deallocator :pointer))
(cffi:defctype buffer :pointer)
;; // Makes a copy of the input and sets an appropriate deallocator.  Useful for
;; // passing in read-only, input protobufs.
;; extern TF_Buffer* TF_NewBufferFromString(const void* proto, size_t proto_len);
(cffi:defcfun ("TF_NewBufferFromString" %TF-NEWBUFFERFROMSTRING) buffer
  (proto :pointer) (proto-len size-t))
;; // Useful for passing *out* a protobuf.
;; extern TF_Buffer* TF_NewBuffer();
(cffi:defcfun ("TF_NewBuffer" %TF-NEWBUFFER) buffer)
;; extern void TF_DeleteBuffer(TF_Buffer*);
(cffi:defcfun ("TF_DeleteBuffer" %TF-DELETEBUFFER) :void (b buffer))

;; extern TF_Buffer TF_GetBuffer(TF_Buffer* buffer);

;; // --------------------------------------------------------------------------
;; // TF_Tensor holds a multi-dimensional array of elements of a single data type.
;; // For all types other than TF_STRING, the data buffer stores elements
;; // in row major order.  E.g. if data is treated as a vector of TF_DataType:
;; //
;; //   element 0:   index (0, ..., 0)
;; //   element 1:   index (0, ..., 1)
;; //   ...
;; //
;; // The format for TF_STRING tensors is:
;; //   start_offset: array[uint64]
;; //   data:         byte[...]
;; //
;; //   The string length (as a varint), followed by the contents of the string
;; //   is encoded at data[start_offset[i]]]. TF_StringEncode and TF_StringDecode
;; //   facilitate this encoding.
(cffi:defctype tensor :pointer)
;; // Return a new tensor that holds the bytes data[0,len-1].
;; //
;; // The data will be deallocated by a subsequent call to TF_DeleteTensor via:
;; //      (*deallocator)(data, len, deallocator_arg)
;; // Clients must provide a custom deallocator function so they can pass in
;; // memory managed by something like numpy.
;; extern TF_Tensor* TF_NewTensor(TF_DataType, const int64_t* dims, int num_dims,
;;                                void* data, size_t len,
;;                                void (*deallocator)(void* data, size_t len,
;;                                                    void* arg),
;;                                void* deallocator_arg);
(cffi:defcfun ("TF_NewTensor" %TF-NEWTENSOR) tensor
  (dt datatype) (dims :pointer) (ndim :int) (data :pointer) (len size-t)
  (deallocator :pointer) (deallocator-args :pointer))
;; // Allocate and return a new Tensor.
;; //
;; // This function is an alternative to TF_NewTensor and should be used when
;; // memory is allocated to pass the Tensor to the C API. The allocated memory
;; // satisfies TensorFlow's memory alignment preferences and should be preferred
;; // over calling malloc and free.
;; //
;; // The caller must set the Tensor values by writing them to the pointer returned
;; // by TF_TensorData with length TF_TensorByteSize.
;; extern TF_Tensor* TF_AllocateTensor(TF_DataType, const int64_t* dims,
;;                                     int num_dims, size_t len);
(cffi:defcfun ("TF_AllocateTensor" %TF-ALLOCATETENSOR) tensor
  (dt datatype) (dims :pointer) (ndim :int) (len size-t))
;; // Destroy a tensor.
;; extern void TF_DeleteTensor(TF_Tensor*);
(cffi:defcfun ("TF_DeleteTensor" %TF-DELETETENSOR) :void (tnsr tensor))
;; // Return the type of a tensor element.
;; extern TF_DataType TF_TensorType(const TF_Tensor*);
(cffi:defcfun ("TF_TensorType" %TF-TENSORTYPE) datatype (tnsr tensor))
;; // Return the number of dimensions that the tensor has.
;; extern int TF_NumDims(const TF_Tensor*);
(cffi:defcfun ("TF_NumDims" %TF-NUMDIMS) :int (tnsr tensor))
;; // Return the length of the tensor in the "dim_index" dimension.
;; // REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
;; extern int64_t TF_Dim(const TF_Tensor* tensor, int dim_index);
(cffi:defcfun ("TF_Dim" %TF-DUM) :int64 (tnsr tensor) (dim-index :int))
;; // Return the size of the underlying data in bytes.
;; extern size_t TF_TensorByteSize(const TF_Tensor*);
(cffi:defcfun ("TF_TensorByteSize" %TF-TENSORBYTESIZE) size-t (tnsr tensor))
;; // Return a pointer to the underlying data buffer.
;; extern void* TF_TensorData(const TF_Tensor*);
(cffi:defcfun ("TF_TensorData" %TF-TENSORDATA) :pointer (tnsr tensor))
