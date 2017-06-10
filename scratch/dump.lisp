;; XXX HERE GOES TEST CODES

(in-package :tf)

(TF_DataTypeSize :TF_DOUBLE)

(let* ((dims (foreign-alloc :int64 :initial-contents '(2 2)))
       (tensor (TF_AllocateTensor :TF_DOUBLE dims 2 1)))
  (format T "~%")
  (format T "NDIMS: ~A~%" (TF_NumDims tensor))
  (format T "DIM: ~A, ~A~%" (TF_Dim tensor 0) (TF_Dim tensor 1))
  (format T "BSIZE: ~A~%" (TF_TensorByteSize tensor))
  (format T "DTYPE: ~A~%" (TF_TensorType tensor))
  (TF_DeleteTensor tensor)
  (foreign-free dims))

(let* ((graph (TF_NewGraph)))
  (print graph)
  (TF_DeleteGraph graph))
