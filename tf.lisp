(in-package :tf)

;; HERE, PUBLIC APIs WILL BE CREATED IF POSSIBLE

(sb-alien:define-alien-routine ("TF_Version" tensorflow-version) sb-alien:c-string)
(sb-alien:define-alien-routine ("TF_DataTypeSize" tensorflow-data-type-size) sb-alien:int
  (datatype sb-alien:int))

(tensorflow-data-type-size 3)

(tensorflow-version)

(defun version () (TF_Version))

(defun data-type-size (datatype) (TF_DataTypeSize datatype))

(defun new-status () (TF_NewStatus))
(defun delete-status (status) (TF_DeleteStatus status))

(defun set-status (status code msg) (TF_SetStatus status code msg))
(defun get-code (status) (TF_GetCode status))
(defun message (status) (TF_Message status))

(defun new-buffer () (TF_NewBuffer))
(defun delete-buffer (buffer) (TF_DeleteBuffer buffer))

(defun new-tensor (datatype dims)
  (let* ((ndim (length dims))
         (len (* ndim (data-type-size datatype))))
    (cffi:with-foreign-object (pdims :int64 ndim)
      (loop :for i :from 0 :below ndim
            :do (setf (cffi:mem-aref pdims :int64 i) (elt dims i)))
      (TF_AllocateTensor datatype pdims ndim len))))
(defun delete-tensor (tensor) (TF_DeleteTensor tensor))

(defun tensor-type (tensor) (TF_TensorType tensor))
(defun tensor-shape (tensor)
  (let ((ndim (TF_NumDims tensor))
        (dims nil))
    (loop :for i :from 0 :below ndim
          :do (push (TF_Dim tensor i) dims))
    (reverse dims)))
(defun tensor-byte-size (tensor) (TF_TensorByteSize tensor))
(defun tensor-data-pointer (tensor) (TF_TensorData tensor))
(defun tensor-data (tensor)
  ;; XXX float should be from TF_FLOAT
  (let* ((nelem (reduce #'* (tensor-shape tensor)))
         (data (make-array nelem :element-type 'single-float))
         (ptr (tensor-data-pointer tensor)))
    (loop :for i :from 0 :below nelem
          :do (setf (aref data i) (cffi:mem-aref ptr :float i)))
    data))

(let ((status (new-status)))
  (set-status status :TF_OK "Hello")
  (print (get-code status))
  (print (message status))
  (delete-status status))

(let ((buffer (new-buffer)))
  (delete-buffer buffer))

(let ((tensor (new-tensor :TF_FLOAT '(2 3))))
  (print (tensor-shape tensor))
  (print (tensor-byte-size tensor))
  (setf (cffi:mem-aref (tensor-data-pointer tensor) :float 0) 123.45)
  (setf (cffi:mem-aref (tensor-data-pointer tensor) :float 3) 123.45)
  (print (tensor-data tensor))
  (delete-tensor tensor))
