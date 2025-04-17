import { render, screen } from '@testing-library/react';
import { LoadingIndicator } from '../LoadingIndicator';

// Mock the FontAwesomeIcon component
jest.mock('@fortawesome/react-fontawesome', () => ({
  FontAwesomeIcon: () => <div data-testid="mock-icon" />
}));

// Mock the loader SVG import
jest.mock('@/assets/loader.svg', () => 'mock-loader-path');

describe('LoadingIndicator Component', () => {
  test('renders loader when isLoading is true', () => {
    render(<LoadingIndicator isLoading={true} />);
    
    const loader = screen.getByRole('img');
    expect(loader).toBeInTheDocument();
    expect(loader).toHaveAttribute('src', 'mock-loader-path');
    expect(loader).toHaveAttribute('alt', 'processing');
    expect(loader).toHaveClass('animate-spin');
    
    // Check that the icon is not rendered
    expect(screen.queryByTestId('mock-icon')).not.toBeInTheDocument();
  });

  test('renders check icon when isLoading is false', () => {
    render(<LoadingIndicator isLoading={false} />);
    
    // Check that the loader is not rendered
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    
    // Check that the icon is rendered
    const icon = screen.getByTestId('mock-icon');
    expect(icon).toBeInTheDocument();
  });
});
