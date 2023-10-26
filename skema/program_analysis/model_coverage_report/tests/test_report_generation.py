from skema.program_analysis.magit.magit import Status

def test_all_valid():
    """Test cast for Status.all_valid static method"""
    status_list_1 = [Status.VALID, Status.VALID, Status.VALID]
    assert Status.all_valid(status_list_1)
    
    status_list_2 = [Status.VALID, Status.EXCEPTION, Status.TIMEOUT]
    assert not Status.all_valid(status_list_2)

def test_overall_status():
    """Test cast for Status.get_overall_status static method"""
    status_list_1 = [Status.VALID, Status.VALID, Status.VALID]
    assert Status.get_overall_status(status_list_1) == Status.VALID
    
    status_list_2 = [Status.VALID, Status.EXCEPTION, Status.TIMEOUT]
    assert Status.get_overall_status(status_list_2) == Status.TIMEOUT
