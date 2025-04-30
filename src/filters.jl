function create_filter(T, grid, cutoff; sigma = 1, filter_type = "sinc", force_cpu = false)
    # TODO extend to multiple dimensions
    N = length(grid)
    N2 = Int(N / 2)
    _kernel = zeros(T, N, N)

    if filter_type == "gaussian"
        k = Int(cutoff)
        center = k / 2
        for i = 1:k
            for j = 1:k
                x = center - i
                y = center - j
                _kernel[i, j] = exp(-(x^2 + y^2) / (2 * sigma^2))
            end
        end
    elseif filter_type == "sinc"
        omega = cutoff * pi
        for x = (-N2+1):1:N2
            for y = (-N2+1):1:N2
                _kernel[x+N2, y+N2] = sinc(x * omega) * sinc(y * omega)
            end
        end
        _kernel = circshift(_kernel, (N2, N2))
    elseif filter_type == "lanczos"
        @warn "You should NOT use lanczos for CNO upsampling because this kernel has a low weight in the orthogonal directions, which is exactly the direction where we create high frequencies with a CNO."
        k = Int(cutoff)
        for i = 1:(2*k+1)
            for j = 1:(2*k+1)
                x = i - (k + 1)
                y = j - (k + 1)
                pos = sqrt(x^2 + y^2)
                _kernel[i, j] = sinc(pos) * sinc(pos / k)
            end
        end
    elseif filter_type == "identity"
        _kernel .= 1
    else
        error("Filter type not recognized")
    end

    # normalize the kernel
    _kernel = _kernel / sum(_kernel)

    # Do the fft of the kernel once
    if CUDA.functional() && !force_cpu
        _kernel = CuArray(_kernel)
    end
    K_f = fft(_kernel, (1, 2))

    function apply_fitler(x)
        # Perform circular convolution using FFT (notice I am assuming PBC in both directions)
        X_f = fft(x, (1, 2))
        filtered_f = X_f .* K_f
        real(ifft(filtered_f, (1, 2)))
    end
end
