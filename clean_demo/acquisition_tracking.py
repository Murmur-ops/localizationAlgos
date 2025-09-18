"""
Realistic Acquisition and Tracking Loops for FTL System
Implements 2D search and code/carrier tracking
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


class TrackingState(Enum):
    """Tracking loop states"""
    SEARCHING = "searching"
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass
class AcquisitionResult:
    """Results from acquisition search"""
    found: bool
    code_phase: float  # Chips
    frequency_offset: float  # Hz
    peak_snr: float  # dB
    correlation_peak: float
    search_time_ms: float


@dataclass
class TrackingMetrics:
    """Tracking loop performance metrics"""
    code_phase_error: float  # Chips
    carrier_phase_error: float  # Radians
    frequency_error: float  # Hz
    loop_snr: float  # dB
    lock_indicator: float  # 0-1
    state: TrackingState


class GoldCodeAcquisition:
    """
    Realistic 2D acquisition search for Gold codes
    Searches over code phase and frequency offset
    """

    def __init__(self,
                 gold_code: np.ndarray,
                 sample_rate: float,
                 chip_rate: float,
                 search_freq_range: float = 10000,  # ±10 kHz
                 freq_bins: int = 41):  # 500 Hz steps

        self.gold_code = gold_code
        self.code_length = len(gold_code)
        self.sample_rate = sample_rate
        self.chip_rate = chip_rate
        self.samples_per_chip = int(sample_rate / chip_rate)

        # Frequency search space
        self.search_freq_range = search_freq_range
        self.freq_bins = freq_bins
        self.freq_steps = np.linspace(-search_freq_range, search_freq_range, freq_bins)

        # Pre-compute reference signals for each frequency bin
        self.reference_signals = self._generate_reference_signals()

        # Threshold for detection (based on false alarm probability)
        self.threshold_factor = 4.0  # Adjust based on Pfa requirements

    def _generate_reference_signals(self) -> dict:
        """Pre-generate reference signals for all frequency bins"""
        references = {}

        # Upsample Gold code to sample rate
        upsampled_code = np.repeat(self.gold_code, self.samples_per_chip)
        signal_length = len(upsampled_code)
        t = np.arange(signal_length) / self.sample_rate

        for freq in self.freq_steps:
            # Apply frequency shift
            freq_shift = np.exp(2j * np.pi * freq * t)
            references[freq] = upsampled_code * freq_shift

        return references

    def acquire(self, received_signal: np.ndarray) -> AcquisitionResult:
        """
        Perform 2D acquisition search

        This is what real GNSS/ranging systems do!
        """
        import time
        start_time = time.time()

        best_metric = 0
        best_code_phase = 0
        best_frequency = 0
        noise_floor = np.inf

        # Ensure signal length matches
        signal_length = min(len(received_signal),
                           len(self.gold_code) * self.samples_per_chip)
        received_signal = received_signal[:signal_length]

        # Search over frequency bins
        for freq in self.freq_steps:
            reference = self.reference_signals[freq][:signal_length]

            # FFT-based circular correlation for all code phases at once
            correlation = np.abs(np.fft.ifft(
                np.fft.fft(received_signal) * np.conj(np.fft.fft(reference))
            ))

            # Find peak
            peak_idx = np.argmax(correlation)
            peak_value = correlation[peak_idx]

            # Estimate noise floor (excluding peak region)
            mask = np.ones(len(correlation), dtype=bool)
            mask[max(0, peak_idx-10):min(len(correlation), peak_idx+10)] = False
            if np.any(mask):
                current_noise = np.median(correlation[mask])
                noise_floor = min(noise_floor, current_noise)

            # Update best if this is stronger
            if peak_value > best_metric:
                best_metric = peak_value
                best_code_phase = peak_idx / self.samples_per_chip
                best_frequency = freq

        # Calculate SNR
        snr_linear = best_metric / noise_floor if noise_floor > 0 else 1
        snr_db = 10 * np.log10(snr_linear)

        # Determine if acquisition successful
        found = best_metric > self.threshold_factor * noise_floor

        # Search time
        search_time_ms = (time.time() - start_time) * 1000

        return AcquisitionResult(
            found=found,
            code_phase=best_code_phase % self.code_length,
            frequency_offset=best_frequency,
            peak_snr=snr_db,
            correlation_peak=best_metric,
            search_time_ms=search_time_ms
        )


class CodeTrackingLoop:
    """
    Delay Lock Loop (DLL) for code tracking
    Uses Early-Prompt-Late correlators
    """

    def __init__(self,
                 gold_code: np.ndarray,
                 sample_rate: float,
                 chip_rate: float,
                 loop_bandwidth: float = 2.0):  # Hz

        self.gold_code = gold_code
        self.code_length = len(gold_code)
        self.sample_rate = sample_rate
        self.chip_rate = chip_rate
        self.samples_per_chip = sample_rate / chip_rate

        # Loop parameters
        self.loop_bandwidth = loop_bandwidth
        self.damping = 0.707  # Critical damping

        # Calculate loop gains
        omega_n = loop_bandwidth * 2 * np.pi
        self.k1 = 2 * self.damping * omega_n
        self.k2 = omega_n ** 2

        # Correlator spacing (chips)
        self.correlator_spacing = 0.5  # Early-Late spacing

        # State variables
        self.code_phase = 0.0
        self.code_rate = chip_rate
        self.phase_error_integral = 0.0

        # Correlator outputs
        self.early = 0
        self.prompt = 0
        self.late = 0

    def update(self, received_signal: np.ndarray, dt: float) -> float:
        """
        Update DLL with new signal samples
        Returns code phase in chips
        """
        # Generate Early, Prompt, Late replicas
        early_phase = (self.code_phase - self.correlator_spacing) % self.code_length
        prompt_phase = self.code_phase % self.code_length
        late_phase = (self.code_phase + self.correlator_spacing) % self.code_length

        # Correlate (simplified - real system would use matched filter)
        signal_len = min(len(received_signal), int(self.code_length * self.samples_per_chip))

        # Generate local codes at different phases
        early_code = self._generate_code_at_phase(early_phase, signal_len)
        prompt_code = self._generate_code_at_phase(prompt_phase, signal_len)
        late_code = self._generate_code_at_phase(late_phase, signal_len)

        # Correlate
        self.early = np.abs(np.sum(received_signal[:signal_len] * np.conj(early_code)))
        self.prompt = np.abs(np.sum(received_signal[:signal_len] * np.conj(prompt_code)))
        self.late = np.abs(np.sum(received_signal[:signal_len] * np.conj(late_code)))

        # Early-minus-late discriminator
        denominator = self.early + self.late
        if denominator > 0:
            discriminator = (self.early - self.late) / denominator
        else:
            discriminator = 0

        # Loop filter (2nd order)
        self.phase_error_integral += discriminator * self.k2 * dt
        code_rate_correction = discriminator * self.k1 + self.phase_error_integral

        # Update code NCO
        self.code_rate = self.chip_rate + code_rate_correction
        self.code_phase += self.code_rate * dt
        self.code_phase = self.code_phase % self.code_length

        return self.code_phase

    def _generate_code_at_phase(self, phase: float, length: int) -> np.ndarray:
        """Generate code replica at specific phase"""
        # Simple nearest-neighbor resampling (real system would interpolate)
        indices = np.arange(length) * self.chip_rate / self.sample_rate + phase
        indices = np.round(indices).astype(int) % self.code_length
        return self.gold_code[indices]

    def get_lock_indicator(self) -> float:
        """
        Calculate lock indicator (0-1)
        Based on ratio of prompt to early+late
        """
        total_power = self.early + self.prompt + self.late
        if total_power > 0:
            return self.prompt / total_power
        return 0


class CarrierTrackingLoop:
    """
    Phase Lock Loop (PLL) for carrier tracking
    2nd order loop with frequency assistance
    """

    def __init__(self,
                 center_frequency: float,
                 sample_rate: float,
                 loop_bandwidth: float = 10.0):  # Hz

        self.center_frequency = center_frequency
        self.sample_rate = sample_rate

        # Loop parameters
        self.loop_bandwidth = loop_bandwidth
        self.damping = 0.707

        # Calculate loop gains
        omega_n = loop_bandwidth * 2 * np.pi
        self.k1 = 2 * self.damping * omega_n
        self.k2 = omega_n ** 2

        # State variables
        self.carrier_phase = 0.0
        self.carrier_frequency = center_frequency
        self.freq_error_integral = 0.0

        # Discriminator outputs
        self.i_prompt = 0
        self.q_prompt = 0

    def update(self, prompt_correlation: complex, dt: float) -> Tuple[float, float]:
        """
        Update PLL with prompt correlation
        Returns (phase, frequency)
        """
        self.i_prompt = np.real(prompt_correlation)
        self.q_prompt = np.imag(prompt_correlation)

        # Atan discriminator (handles ±180° range)
        if self.i_prompt != 0:
            phase_error = np.arctan(self.q_prompt / self.i_prompt)
        else:
            phase_error = np.pi/2 * np.sign(self.q_prompt)

        # Loop filter
        self.freq_error_integral += phase_error * self.k2 * dt
        freq_correction = phase_error * self.k1 + self.freq_error_integral

        # Update carrier NCO
        self.carrier_frequency = self.center_frequency + freq_correction
        self.carrier_phase += 2 * np.pi * self.carrier_frequency * dt
        self.carrier_phase = self.carrier_phase % (2 * np.pi)

        return self.carrier_phase, self.carrier_frequency

    def get_lock_indicator(self) -> float:
        """
        Calculate PLL lock indicator
        Based on I vs Q power ratio
        """
        total_power = self.i_prompt**2 + self.q_prompt**2
        if total_power > 0:
            return abs(self.i_prompt) / np.sqrt(total_power)
        return 0


class IntegratedTrackingLoop:
    """
    Complete tracking system combining acquisition, DLL, and PLL
    This is what real GNSS receivers use!
    """

    def __init__(self,
                 gold_code: np.ndarray,
                 sample_rate: float = 10e6,
                 chip_rate: float = 1.023e6,
                 carrier_frequency: float = 2.4e9):

        self.gold_code = gold_code
        self.sample_rate = sample_rate
        self.chip_rate = chip_rate
        self.carrier_frequency = carrier_frequency

        # Initialize components
        self.acquisition = GoldCodeAcquisition(
            gold_code, sample_rate, chip_rate
        )

        self.dll = CodeTrackingLoop(
            gold_code, sample_rate, chip_rate, loop_bandwidth=2.0
        )

        self.pll = CarrierTrackingLoop(
            carrier_frequency, sample_rate, loop_bandwidth=10.0
        )

        # State machine
        self.state = TrackingState.SEARCHING
        self.lock_counter = 0
        self.loss_counter = 0

        # Thresholds
        self.lock_threshold = 10  # Consecutive good measurements to declare lock
        self.loss_threshold = 5   # Consecutive bad measurements to declare loss

    def process(self, received_signal: np.ndarray, dt: float) -> TrackingMetrics:
        """
        Process received signal through complete tracking system
        """

        if self.state == TrackingState.SEARCHING:
            # Perform acquisition
            result = self.acquisition.acquire(received_signal)

            if result.found:
                # Initialize tracking loops
                self.dll.code_phase = result.code_phase
                self.pll.carrier_frequency = self.carrier_frequency + result.frequency_offset
                self.state = TrackingState.ACQUIRING
                self.lock_counter = 0

                return TrackingMetrics(
                    code_phase_error=0,
                    carrier_phase_error=0,
                    frequency_error=result.frequency_offset,
                    loop_snr=result.peak_snr,
                    lock_indicator=0.5,
                    state=self.state
                )
            else:
                return TrackingMetrics(
                    code_phase_error=float('inf'),
                    carrier_phase_error=float('inf'),
                    frequency_error=float('inf'),
                    loop_snr=-10,
                    lock_indicator=0,
                    state=self.state
                )

        elif self.state == TrackingState.ACQUIRING or self.state == TrackingState.TRACKING:
            # Update tracking loops
            code_phase = self.dll.update(received_signal, dt)

            # Get prompt correlation for PLL
            prompt_idx = int(code_phase * self.dll.samples_per_chip)
            if prompt_idx < len(received_signal):
                prompt_sample = received_signal[prompt_idx]
            else:
                prompt_sample = 0

            carrier_phase, carrier_freq = self.pll.update(prompt_sample, dt)

            # Calculate lock indicators
            dll_lock = self.dll.get_lock_indicator()
            pll_lock = self.pll.get_lock_indicator()
            combined_lock = (dll_lock + pll_lock) / 2

            # Update state machine
            if combined_lock > 0.8:
                self.lock_counter += 1
                self.loss_counter = 0

                if self.lock_counter > self.lock_threshold:
                    self.state = TrackingState.TRACKING
            else:
                self.loss_counter += 1
                self.lock_counter = 0

                if self.loss_counter > self.loss_threshold:
                    self.state = TrackingState.LOST

            # Calculate metrics
            snr_estimate = 20 * np.log10(self.dll.prompt / (self.dll.early + self.dll.late + 1e-10))

            return TrackingMetrics(
                code_phase_error=(self.dll.early - self.dll.late) / (self.dll.early + self.dll.late + 1e-10),
                carrier_phase_error=np.arctan2(self.pll.q_prompt, self.pll.i_prompt),
                frequency_error=carrier_freq - self.carrier_frequency,
                loop_snr=snr_estimate,
                lock_indicator=combined_lock,
                state=self.state
            )

        else:  # LOST state
            self.state = TrackingState.SEARCHING
            return self.process(received_signal, dt)


def test_acquisition_tracking():
    """Test the acquisition and tracking system"""
    print("REALISTIC ACQUISITION & TRACKING TEST")
    print("="*60)

    # Generate test Gold code
    np.random.seed(42)
    gold_code = np.random.choice([-1, 1], size=127)

    # System parameters
    sample_rate = 10e6  # 10 MHz
    chip_rate = 1.023e6  # 1.023 MHz
    carrier_freq = 2.4e9  # 2.4 GHz

    # Create tracking system
    tracker = IntegratedTrackingLoop(
        gold_code, sample_rate, chip_rate, carrier_freq
    )

    print("\n1. ACQUISITION PHASE:")
    print("-"*40)

    # Generate test signal with known parameters
    true_code_phase = 50  # chips
    true_freq_offset = 3500  # Hz
    snr_db = 15

    # Create test signal
    samples_per_chip = int(sample_rate / chip_rate)
    signal_length = len(gold_code) * samples_per_chip
    t = np.arange(signal_length) / sample_rate

    # Generate signal with code and frequency offset
    upsampled_code = np.repeat(gold_code, samples_per_chip)
    shifted_code = np.roll(upsampled_code, int(true_code_phase * samples_per_chip))
    freq_shift = np.exp(2j * np.pi * true_freq_offset * t)

    # Add noise
    signal_power = np.mean(np.abs(shifted_code)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(signal_length) +
                                       1j * np.random.randn(signal_length))

    test_signal = shifted_code * freq_shift + noise

    # Perform acquisition
    acq_result = tracker.acquisition.acquire(test_signal)

    print(f"True code phase: {true_code_phase:.1f} chips")
    print(f"Est. code phase: {acq_result.code_phase:.1f} chips")
    print(f"Code phase error: {abs(acq_result.code_phase - true_code_phase):.2f} chips")
    print(f"\nTrue frequency: {true_freq_offset:.0f} Hz")
    print(f"Est. frequency: {acq_result.frequency_offset:.0f} Hz")
    print(f"Frequency error: {abs(acq_result.frequency_offset - true_freq_offset):.0f} Hz")
    print(f"\nAcquisition SNR: {acq_result.peak_snr:.1f} dB")
    print(f"Search time: {acq_result.search_time_ms:.1f} ms")
    print(f"Acquisition: {'SUCCESS' if acq_result.found else 'FAILED'}")

    print("\n2. TRACKING PHASE:")
    print("-"*40)

    # Initialize tracking at acquisition point
    tracker.dll.code_phase = acq_result.code_phase
    tracker.pll.carrier_frequency = carrier_freq + acq_result.frequency_offset
    tracker.state = TrackingState.ACQUIRING

    # Run tracking for multiple epochs
    dt = 0.001  # 1ms update
    tracking_results = []

    for epoch in range(20):
        # Generate new signal segment (simulating continuous reception)
        metrics = tracker.process(test_signal, dt)
        tracking_results.append(metrics)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: State={metrics.state.value:9s}, "
                  f"Lock={metrics.lock_indicator:.2f}, "
                  f"SNR={metrics.loop_snr:.1f}dB")

    # Check final state
    final_state = tracking_results[-1].state
    print(f"\nFinal tracking state: {final_state.value}")

    if final_state == TrackingState.TRACKING:
        print("✓ Successfully acquired and tracking!")
    else:
        print("✗ Tracking not achieved")

    print("\n3. KEY FEATURES IMPLEMENTED:")
    print("-"*40)
    print("✓ 2D acquisition search (code × frequency)")
    print("✓ FFT-based parallel correlation")
    print("✓ Early-Prompt-Late DLL for code tracking")
    print("✓ 2nd order PLL for carrier tracking")
    print("✓ State machine (SEARCHING → ACQUIRING → TRACKING)")
    print("✓ Lock indicators and quality metrics")
    print("✓ Realistic loop bandwidths and damping")


if __name__ == "__main__":
    test_acquisition_tracking()