import { render, screen, fireEvent } from '@testing-library/react';
import { NewChatModal } from '../NewChatModal';
import { useCreateChat } from '@/api-state/chat-messages/hooks';
import { useNavigate } from 'react-router-dom';
import { generateChatRoute } from '@/pages/routes';

// Mock the dependencies
jest.mock('@/api-state/chat-messages/hooks', () => ({
  useCreateChat: jest.fn(),
}));

jest.mock('@/pages/routes', () => ({
  generateChatRoute: jest.fn(),
}));

// Mock FontAwesomeIcon
jest.mock('@fortawesome/react-fontawesome', () => ({
  FontAwesomeIcon: () => <div data-testid="mock-icon" />
}));

// Mock lucide-react
jest.mock('lucide-react', () => ({
  XIcon: () => <div data-testid="x-icon" />
}));

describe('NewChatModal Component', () => {
  let mockCreateChat: jest.Mock;
  let mockNavigate: jest.Mock;

  // Setup default mocks before each test
  beforeEach(() => {
    // Mock the createChat mutation
    mockCreateChat = jest.fn();
    (useCreateChat as jest.Mock).mockReturnValue({
      mutate: mockCreateChat,
      isPending: false,
    });

    // Mock the navigate function
    mockNavigate = jest.fn();
    (useNavigate as jest.Mock).mockReturnValue(mockNavigate);

    // Mock the generateChatRoute function
    (generateChatRoute as jest.Mock).mockImplementation((id) => `/chat/${id}`);
  });

  test('renders the trigger button correctly', () => {
    render(<NewChatModal />);
    const newChatButton = screen.getByRole('button', { name: /new chat/i });
    expect(newChatButton).toBeInTheDocument();
    expect(screen.getByTestId('mock-icon')).toBeInTheDocument();
  });

  test('shows loading state when creating a chat', () => {
    // Set isPending to true for loading state
    (useCreateChat as jest.Mock).mockReturnValue({
      mutate: mockCreateChat,
      isPending: true,
    });

    render(<NewChatModal />);

    // Find and click the New Chat button to open the dialog
    const triggerButton = screen.getByRole('button', { name: /new chat/i });
    fireEvent.click(triggerButton);

    // Now the dialog content should have a "Creating..." button
    const createButton = screen.getByRole('button', { name: /creating/i });
    expect(createButton).toBeInTheDocument();
    expect(createButton).toBeDisabled();
  });
});
